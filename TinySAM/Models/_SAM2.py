from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
from tqdm.auto import trange

import numpy as np
import time
import torch._dynamo
torch._dynamo.config.suppress_errors = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SAM2:
    def __init__(self, model_name: str = "facebook/sam2-hiera-large", device: str = None, compile=False):
        '''
        Segment Anything Model 2 (SAM2) model for zero-shot object detection.
        
        Args:
            model_name (str): model name from Hugging Face Model Hub. (Default: "facebook/sam2-hiera-large", see hugging face model hub for more model variations)
            device (str): device to run the model on. (Default: None, Auto Device Selection)
        '''
        
        self.device = device
        
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        global DEVICE 
        DEVICE = self.device
        
        self.model = SAM2ImagePredictor.from_pretrained(model_name, device_map = self.device)
        
        if compile:
            self.model.model.compile()
        
        self.image_size = self.model.model.image_size
        
        self.img_encoder_isolated = self.model.model.image_encoder
        self.prompt_encoder_isolated = self.model.model.sam_prompt_encoder
        self.mask_decoder_isolated = self.model.model.sam_mask_decoder
    
    @torch.inference_mode()
    @torch.autocast(device_type=DEVICE, dtype=torch.bfloat16)
    @torch.no_grad()
    def __call__(self, images, boxes):
        masks = []
        for i in trange(len(images)):
            if len(boxes[i]) == 0:
                masks.append([])
                continue
            self.model.set_image(images[i])
            masks_, _, _ = self.model.predict(box=boxes[i], multimask_output=False)
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
        
        self.model._orig_hw = [(self.image_size, self.image_size)]
        self.model._is_image_set = True
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
        
        backbone_out = self.model.model.forward_image(img)
        _, vision_feats, _, _ = self.model.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self.model._bb_feat_sizes[::-1])
        ][::-1]
        self.model._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self.model._is_image_set = True
        
        masks_, _, _ = self.model.predict(box=queries, multimask_output=False)
        return masks_