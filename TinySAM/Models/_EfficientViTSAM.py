# from segment_anything_hq import sam_model_registry, SamPredictor
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
import torch
from tqdm.auto import trange

import numpy as np
import time
import torch._dynamo
torch._dynamo.config.suppress_errors = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class EfficientViTSAM:
    def __init__(self, variant = "efficientvit-sam-xl1", device: str = None, compile=False):
        '''
        EfficientViTSAM model for zero-shot object detection.
        
        Args:
            variant (str): variant of the model to use. (Default: "efficientvit-sam-xl1", one of "efficientvit-sam-xl1", ""efficientvit-sam-l0", "efficientvit-sam-l1", "efficientvit-sam-l2", "efficientvit-sam-xl0")
            device (str): device to run the model on. (Default: None, Auto Device Selection)
        '''
        
        self.device = device
        
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        global DEVICE 
        DEVICE = self.device
        
        chk = variant.replace('-','_')
        self.model = create_efficientvit_sam_model(variant, pretrained=True, weight_url=f'weights/{chk}.pt')
        self.model.to(self.device)
        self.model.eval()
        
        self.predictor = EfficientViTSamPredictor(self.model)
        
        if compile:
            self.model.compile()
        
        self.image_size = self.model.image_size
        
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
        
        sample_input_image = torch.rand((1, 3, self.image_size[0], self.image_size[1])).to(self.device)
        sample_box_queries = torch.randint(low=0, high=np.min(self.image_size), size=(n_boxes, 4)).to(self.device)
        
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