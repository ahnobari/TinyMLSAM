from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
import torch
from tqdm.auto import trange

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