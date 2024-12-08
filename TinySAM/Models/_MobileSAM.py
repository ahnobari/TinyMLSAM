from mobile_sam import sam_model_registry, SamPredictor
import torch
from tqdm.auto import trange

import numpy as np
import time
import torch._dynamo
torch._dynamo.config.suppress_errors = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MobileSAM:
    def __init__(self, variant:str = "", device: str = None, compile=False):
        '''
        MobileSAM model for zero-shot object detection.
        
        Args:
            device (str): device to run the model on. (Default: None, Auto Device Selection)
        '''
        
        self.device = device
        
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        global DEVICE 
        DEVICE = self.device
        
        self.model = sam_model_registry['vit_t'](checkpoint='weights/mobile_sam.pt')
        self.model.to(self.device)
        self.model.eval()
        
        self.predictor = SamPredictor(self.model)
        
        if compile:
            self.model.compile()
        
        self.image_size = self.model.image_encoder.img_size
        
        self.img_encoder_isolated = self.model.image_encoder
        self.prompt_encoder_isolated = self.model.prompt_encoder
        self.mask_decoder_isolated = self.model.mask_decoder
    
    @torch.inference_mode()
    @torch.autocast(device_type=DEVICE, dtype=torch.bfloat16)
    @torch.no_grad()
    def __call__(self, images, boxes):
        masks = []
        for i in trange(len(images)):
            if len(boxes[i]) == 0:
                masks.append([])
                continue
            # self.model.set_image(images[i])
            self.predictor.set_image(images[i])
            box_input = self.predictor.transform.apply_boxes(boxes[i], self.predictor.original_size)
            masks_, _, _ = self.predictor.predict_torch(None, None, boxes=torch.tensor(box_input).to(self.device).float(), multimask_output=False)
            masks_ = masks_.cpu().numpy()
            if masks_.ndim == 3:
                masks.append(masks_.astype(bool))
            else:
                masks.append(masks_.squeeze(1).astype(bool))
            
        return masks
    
    def run_throughput_test(self, n_boxes = 28, n_exprs = 100):
        # make sure all the layers are compiled
        self.img_encoder_isolated.compile()
        self.prompt_encoder_isolated.compile()
        self.mask_decoder_isolated.compile()
        
        sample_input_image = torch.rand((1, 3, self.image_size, self.image_size)).to(self.device)
        sample_box_queries = torch.randint(low=0, high=self.image_size, size=(n_boxes, 4)).to(self.device)
        
        # sort box coords
        sample_box_queries = sample_box_queries.sort(dim=1).values.float()
        
        # compilation run
        self.raw_call(sample_input_image, sample_box_queries)
        
        times = []
        
        if self.device == "cuda":
            for i in trange(n_exprs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                self.raw_call(sample_input_image, sample_box_queries)
                end.record()

                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
        
        else:
            for i in trange(n_exprs):
                start = time.time()
                self.raw_call(sample_input_image, sample_box_queries)
                end = time.time()
                times.append((end-start)*1000)
            
        print(f"Throughput Test: {n_exprs} runs of {n_boxes} boxes")
        print(f"Mean Time: {np.mean(times)}  +/- {np.std(times)} ms")
        
    
    @torch.inference_mode()
    @torch.autocast(device_type=DEVICE, dtype=torch.bfloat16)
    @torch.no_grad()
    def raw_call(self, img, queries):
        img_emb = self.img_encoder_isolated(img)
        sparse_embeddings, dense_embeddings = self.prompt_encoder_isolated(points=None, masks = None, boxes=queries)
        low_res_masks, iou_predictions = self.mask_decoder_isolated(
            image_embeddings=img_emb,
            image_pe=self.prompt_encoder_isolated.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )
        
        return low_res_masks, iou_predictions