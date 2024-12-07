from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
from tqdm.auto import trange

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