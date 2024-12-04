from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
from tqdm.auto import trange

class SAM2:
    def __init__(self, model_name: str = "facebook/sam2-hiera-large", device: str = None, compile=False):
        '''
        Grounding DINO model for zero-shot object detection. Prior for SAM.
        
        Args:
            model_name (str): model name from Hugging Face Model Hub. (Default: "IDEA-Research/grounding-dino-base", Tiny Variant: "IDEA-Research/grounding-dino-tiny")
            device (str): device to run the model on. (Default: None, Auto Device Selection)
        '''
        
        self.device = device
        
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = SAM2ImagePredictor.from_pretrained(model_name, device_map = self.device)
        
        if compile:
            self.model.model.compile()
    
    @torch.inference_mode()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def __call__(self, images, boxes):
        masks = []
        for i in trange(len(images)):
            self.model.set_image(images[i])
            masks_, _, _ = self.model.predict(box=boxes[i], multimask_output=False)
            masks.append(masks_.squeeze(1).astype(bool))
            
        return masks