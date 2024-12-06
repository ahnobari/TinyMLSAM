from mobile_sam import sam_model_registry, SamPredictor
import torch
from tqdm.auto import trange

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MobileSAM:
    def __init__(self, device: str = None, compile=False):
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
            self.model.model.compile()
    
    @torch.inference_mode()
    @torch.autocast(device_type=DEVICE, dtype=torch.bfloat16)
    @torch.no_grad()
    def __call__(self, images, boxes):
        masks = []
        for i in trange(len(images)):
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