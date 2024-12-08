# fake the edgesam unused module
class projects(object):
    class efficientdet(object):
        def __init__(self, *args, **kwargs):
            pass
            
import sys
sys.modules["projects.EfficientDet"] = projects

from edge_sam import sam_model_registry, SamPredictor
import torch
from tqdm.auto import trange


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class EdgeSAM:
    def __init__(self, variant = "", device: str = None, compile=False):
        '''
        EdgeSAM model for zero-shot object detection.
        
        Args:
            variant (str): variant of the model to use. (Default: "", one of "", "_3x")
            device (str): device to run the model on. (Default: None, Auto Device Selection)
        '''
        
        self.device = device
        
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        global DEVICE 
        DEVICE = self.device
        
        if variant == "Base":
            variant = ""
        
        self.model = sam_model_registry["edge_sam"](checkpoint=f'weights/edge_sam{variant}.pth')
        self.model.to(self.device)
        self.model.eval()
        
        self.predictor = SamPredictor(self.model)
        
        if compile:
            self.model.compile()
            
        self.image_size = self.model.image_encoder.img_size
        
        self.img_encoder_isolated = self.model.image_encoder
        self.prompt_encoder_isolated = self.model.prompt_encoder
        self.mask_decoder_isolated = self.model.mask_decoder
        
        self.sample_input_image = None
        self.sample_box_queries = None
    
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
            masks_, _, _ = self.predictor.predict_torch(None, None, None, boxes=torch.tensor(box_input).to(self.device).float())
            masks_ = masks_.cpu().numpy()[:, 0:1, :, :]
            if masks_.ndim == 3:
                masks.append(masks_.astype(bool))
            else:
                masks.append(masks_.squeeze(1).astype(bool))
            
        return masks
        