import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from torch.utils.data import DataLoader
from typing import List
from tqdm.auto import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class GDino:
    def __init__(self, model_name: str = "IDEA-Research/grounding-dino-base", device: str = None, box_threshold: float = 0.3, text_threshold: float = 0.25, compile=False):
        '''
        Grounding DINO model for zero-shot object detection. Prior for SAM.
        
        Args:
            model_name (str): model name from Hugging Face Model Hub. (Default: "IDEA-Research/grounding-dino-base", Tiny Variant: "IDEA-Research/grounding-dino-tiny")
            device (str): device to run the model on. (Default: None, Auto Device Selection)
        '''
        
        self.device = device
        
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        global DEVICE 
        DEVICE = self.device
        
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name, device_map = self.device)
        self.model.eval()
        if compile:
            self.model.compile()
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    @torch.inference_mode()
    @torch.autocast(device_type=DEVICE, dtype=torch.bfloat16)
    @torch.no_grad()
    def __call__(self, images: torch.tensor, input_ids: torch.Tensor, target_image_size: tuple, **kwargs):
        batch_size = images.shape[0]
        
        if input_ids.shape[0] != batch_size:
            input_ids = input_ids.repeat(batch_size, 1)
            
        if images.device != self.device:
            images = images.to(self.device)
        
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)
        
        outputs = self.model(pixel_values = images, input_ids = input_ids, **kwargs)
        
        output = self.processor.post_process_grounded_object_detection(outputs, input_ids=input_ids, box_threshold=self.box_threshold, text_threshold=self.text_threshold, target_sizes=[target_image_size]*batch_size)
        
        return output
    
    def run_loader(self, dataloader:  DataLoader, input_ids: torch.Tensor, text_prompts: list[str], target_image_size: tuple, **kwargs):
        
        boxes = []
        labels = []
        scores = []
        
        for images in tqdm(dataloader):
            images = images.to(self.device)
            outputs = self(images, input_ids, target_image_size, **kwargs)
            for output in outputs:
                bs = output['boxes'].cpu().numpy()
                ls = output['labels']
                ss = output['scores'].cpu().numpy()
                valids = []
                for i in range(len(ls)):
                    # only pick first label for any dual predictions
                    if ls[i] + '.' not in text_prompts:
                        ls[i] = ls[i].split(' ')[0]
                        if ls[i] + '.' not in text_prompts:
                            # Remove this prediction if it is not in the text prompts
                            ls[i] = None
                    if ls[i] is not None:
                        valids.append(i)
                boxes.append(bs[valids])
                scores.append(ss[valids])
                labels.append([ls[i] for i in valids])
                
        return boxes, labels, scores