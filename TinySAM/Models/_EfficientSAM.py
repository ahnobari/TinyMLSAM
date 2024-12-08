from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
import torch
from tqdm.auto import trange

import numpy as np
import time
import torch._dynamo
torch._dynamo.config.suppress_errors = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class EfficientSAM:
    def __init__(self, model_variant: str = "S", device: str = None, compile=False):
        '''
        EfficientSAM model for zero-shot object detection.
        
        Args:
            model_name (str): model name from Hugging Face Model Hub. (Default: "S", Tiny Variant: "Ti")
            device (str): device to run the model on. (Default: None, Auto Device Selection)
        '''
        
        self.device = device
        
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        global DEVICE 
        DEVICE = self.device
        
        self.model = build_efficient_sam_vits() if model_variant == "S" else build_efficient_sam_vitt()
        self.model.to(self.device)
        self.model.eval()
        
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
            box_input = torch.tensor(boxes[i], device=self.device).float()
            N_query = box_input.size(0)
            box_input = box_input.view(1, N_query, 2, 2)
            point_labels = torch.zeros(1, N_query, 2, device=self.device).float()
            point_labels[:,:,0] = 2
            point_labels[:,:,1] = 3
            image = torch.tensor(images[i], device=self.device).float()/255.0
            image = image.unsqueeze(0).permute(0, 3, 1, 2)
            
            predicted_logits, predicted_iou = self.model(image, box_input, point_labels)
            
            sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
            predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
            predicted_logits = torch.take_along_dim(predicted_logits, sorted_ids[..., None, None], dim=2)
            
            mask = torch.ge(predicted_logits[0,:,0,:,:], 0).cpu().numpy().astype(bool)
            masks.append(mask)
        return masks
    
    def run_throughput_test(self, n_boxes = 28, n_exprs = 100):
        # make sure all the layers are compiled
        self.img_encoder_isolated.compile()
        self.prompt_encoder_isolated.compile()
        self.mask_decoder_isolated.compile()
        
        sample_input_image = torch.rand((1, 3, self.image_size, self.image_size)).to(self.device)
        sample_box_queries = torch.randint(low=0, high=self.image_size, size=(n_boxes, 4)).to(self.device)
        
        point_labels = torch.zeros(1, n_boxes, 2, device=self.device).float()
        point_labels[:,:,0] = 2
        point_labels[:,:,1] = 3
        
        # sort box coords
        sample_box_queries = sample_box_queries.sort(dim=1).values.float()
        sample_box_queries = sample_box_queries.view(1, n_boxes, 2, 2)
        
        # compilation run
        self.raw_call(sample_input_image, sample_box_queries, point_labels)
        
        times = []
        
        if self.device == "cuda":
            for i in trange(n_exprs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                self.raw_call(sample_input_image, sample_box_queries, point_labels)
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
    def raw_call(self, img, queries, points_labels):
        img_emb = self.img_encoder_isolated(img)
        output_masks, iou_predictions = self.model.predict_masks(
            img_emb,
            queries,
            points_labels,
            multimask_output=True,
            input_h=self.image_size,
            input_w=self.image_size,
            output_h=-1,
            output_w=-1,
        )
        
        return output_masks, iou_predictions