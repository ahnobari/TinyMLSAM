# from segment_anything_hq import sam_model_registry, SamPredictor
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
import torch
from tqdm.auto import trange


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