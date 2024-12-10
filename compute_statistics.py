from TinySAM import *
import TinySAM
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm, trange
from torch.utils.data import DataLoader
import argparse
import pickle
import os
import uuid
from torchinfo import summary
import torch.nn as nn
import json

argparser = argparse.ArgumentParser()
argparser.add_argument('--model', type=str, default='SAM2', help='Model to use for segementation. Default is SAM2. Options: SAM2, EdgeSAM, EfficientSAM, EfficientViTSAM, MobileSAM, SAMHQ')
argparser.add_argument('--model_variant', type=str, default='facebook/sam2.1-hiera-large', help='Model variant to use for segmentation. Default is facebook/sam2.1-hiera-large. Will depend on the model chosen.')
argparser.add_argument('--data_path', type=str, default='Data/cityscapes', help='Path to the dataset. Default is Data/cityscapes.')
argparser.add_argument('--queries_path', type=str, default='results/instances_Base_cityscapes_BT_0.2_TT_0.15_PE.pkl', help='Path to the grounding instances. Default is results/instances_Base_cityscapes_BT_0.2_TT_0.15.pkl.')
argparser.add_argument('--save_masks', action='store_true', help='Save the segmentation masks. Default is False.')
argparser.add_argument('--results_path', type=str, default='results', help='path to save the results (if save_masks is passed the raw results will also be saved). Default is results. File name will include run information and a unique id.')
argparser.add_argument('--use_prompt_engineering', action='store_true', help='Use prompt engineering for grounding. Default is False.')
args = argparser.parse_args()

class SAM2ForwardWrapper(nn.Module):
    def __init__(self, core_model):
        super().__init__()
        self.core_model = core_model
        
    def forward(self, x):
        # x is expected to be (B, C, H, W), a standard image tensor
        # For the sake of summary, we just run part of the model pipeline.
        #
        # If `core_model.forward()` is not implemented for inference,
        # try using a known available method like `forward_image()` 
        # that returns intermediate features. You must return a tensor.
        
        # Example: If `core_model` has `forward_image()` returning features:
        backbone_out = self.core_model.forward_image(x)
        
        # backbone_out is likely a tuple or some features. We need to return a tensor.
        # If it's a tuple of tensors, return one of them:
        if isinstance(backbone_out, tuple) and len(backbone_out) > 0:
            return backbone_out[0]
        elif isinstance(backbone_out, torch.Tensor):
            return backbone_out
        else:
            # If backbone_out is not suitable, create a dummy tensor:
            return torch.zeros((x.size(0), 1), device=x.device)


class EdgeSAMForwardWrapper(nn.Module):
    def __init__(self, edge_sam_instance):
        super().__init__()
        self.img_encoder = edge_sam_instance.img_encoder_isolated
        self.prompt_encoder = edge_sam_instance.prompt_encoder_isolated
        self.mask_decoder = edge_sam_instance.mask_decoder_isolated
        self.image_size = edge_sam_instance.image_size
        self.device = edge_sam_instance.device

    def forward(self, img, queries):
        img_emb = self.img_encoder(img)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(points=None, masks=None, boxes=queries)
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=img_emb,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            num_multimask_outputs=1
        )
        # Return low_res_masks (a tensor) for MAC computation
        return low_res_masks

# Wrapper class for EfficientSAM
class EfficientSAMForwardWrapper(nn.Module):
    def __init__(self, eff_sam_instance):
        super().__init__()
        self.model = eff_sam_instance.model
        self.image_size = eff_sam_instance.image_size
        self.device = eff_sam_instance.device

    def forward(self, img, box_input, point_labels):
        # Model expects: image, box_input (reshaped), and point_labels
        # These should match the logic in eff_sam_instance.__call__
        # where it does something like:
        # predicted_logits, predicted_iou = self.model(image, box_input, point_labels)
        predicted_logits, predicted_iou = self.model(img, box_input, point_labels)
        
        # Return a tensor for summary to analyze.
        # predicted_logits: (B, N, 3, H, W) or similar
        # We can just return predicted_logits as it's a tensor.
        return predicted_logits

# Wrapper class for EfficientViTSAM
class EfficientViTSAMForwardWrapper(nn.Module):
    def __init__(self, eff_vit_sam_instance):
        super().__init__()
        self.img_encoder = eff_vit_sam_instance.img_encoder_isolated
        self.prompt_encoder = eff_vit_sam_instance.prompt_encoder_isolated
        self.mask_decoder = eff_vit_sam_instance.mask_decoder_isolated
        self.image_size = eff_vit_sam_instance.image_size
        self.device = eff_vit_sam_instance.device

    def forward(self, img, queries):
        # Similar to raw_call in EfficientViTSAM
        img_emb = self.img_encoder(img)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(points=None, masks=None, boxes=queries)
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=img_emb,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )
        # Return a tensor (low_res_masks) for MAC computation
        return low_res_masks

# Wrapper class for MobileSAM
class MobileSAMForwardWrapper(nn.Module):
    def __init__(self, mobile_sam_instance):
        super().__init__()
        self.img_encoder = mobile_sam_instance.img_encoder_isolated
        self.prompt_encoder = mobile_sam_instance.prompt_encoder_isolated
        self.mask_decoder = mobile_sam_instance.mask_decoder_isolated
        self.image_size = mobile_sam_instance.image_size
        self.device = mobile_sam_instance.device

    def forward(self, img, queries):
        # Mimic raw_call logic from MobileSAM
        img_emb = self.img_encoder(img)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(points=None, masks=None, boxes=queries)
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=img_emb,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )
        return low_res_masks

class SAMHQForwardWrapper(nn.Module):
    def __init__(self, samhq_instance):
        super().__init__()
        self.img_encoder = samhq_instance.img_encoder_isolated
        self.prompt_encoder = samhq_instance.prompt_encoder_isolated
        self.mask_decoder = samhq_instance.mask_decoder_isolated
        self.image_size = samhq_instance.image_size
        self.device = samhq_instance.device

    def forward(self, img, queries):
        # Mimics raw_call from SAMHQ
        features, interm_features = self.img_encoder(img)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(points=None, masks=None, boxes=queries)
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            hq_token_only=False,
            multimask_output=False,
            interm_embeddings=interm_features
        )
        return low_res_masks

MODELS_AND_VARIANTS = {
    "SAM2": "facebook/sam2.1-hiera-tiny facebook/sam2.1-hiera-small facebook/sam2.1-hiera-large facebook/sam2.1-hiera-base-plus".split(),
    "EdgeSAM": "Base _3x".split(),
    "EfficientSAM": "S Ti".split(),
    "EfficientViTSAM": "efficientvit-sam-xl1 efficientvit-sam-l0 efficientvit-sam-l1 efficientvit-sam-l2 efficientvit-sam-xl0".split(),
    "MobileSAM": ["MobileSAM"],
    "SAMHQ": "vit_h vit_l vit_b vit_tiny".split()
}

results = {}
for model, variants in MODELS_AND_VARIANTS.items():
    results[model] = {}
    for variant in variants:
        print(f"Model: {model}, Variant: {variant}")
        segmentation_model = getattr(TinySAM, model)(variant)
        print(f'Image size: {segmentation_model.image_size}')

        if model == 'SAM2':
            # SAM2 has segmentation_model.model.model
            segmentation_model.model.model.eval()
            wrapped_model = SAM2ForwardWrapper(segmentation_model.model.model).eval()
            
            # input_size = (1, 3, 1024, 2048)
            input_size = (1, 3, 1024, 1024)
            model_stats = summary(wrapped_model, input_size=input_size, col_names=("input_size", "output_size", "num_params", "mult_adds"))
        
        elif model == 'EdgeSAM':
            segmentation_model.model.eval()
            wrapped_model = EdgeSAMForwardWrapper(segmentation_model).eval()
            # dummy_img = torch.randn(1, 3, 1024, 2048, device=segmentation_model.device)
            dummy_img = torch.randn(1, 3, 1024, 1024, device=segmentation_model.device)
            dummy_queries = torch.tensor([[0, 0, 50, 50]], device=segmentation_model.device).float()
            model_stats = summary(
                wrapped_model, 
                input_data=(dummy_img, dummy_queries), 
                col_names=("input_size", "output_size", "num_params", "mult_adds")
            )
        elif model == 'EfficientSAM':
            segmentation_model.model.eval()
            wrapped_model = EfficientSAMForwardWrapper(segmentation_model).eval()

            # Create dummy inputs for EfficientSAM
            dummy_img = torch.randn(1, 3, segmentation_model.image_size, segmentation_model.image_size, device=segmentation_model.device)
            # dummy_img = torch.randn(1, 3, 1024, 2048, device=segmentation_model.device)
            
            n_boxes = 5
            dummy_box_queries = torch.randint(low=0, high=segmentation_model.image_size, size=(n_boxes, 4), device=segmentation_model.device)
            dummy_box_queries = dummy_box_queries.sort(dim=1).values.float()
            dummy_box_queries = dummy_box_queries.view(1, n_boxes, 2, 2)

            dummy_point_labels = torch.zeros(1, n_boxes, 2, device=segmentation_model.device).float()
            dummy_point_labels[:, :, 0] = 2
            dummy_point_labels[:, :, 1] = 3

            model_stats = summary(
                wrapped_model,
                input_data=(dummy_img, dummy_box_queries, dummy_point_labels),
                col_names=("input_size", "output_size", "num_params", "mult_adds")
            )
        elif model == 'EfficientViTSAM':
            segmentation_model.model.eval()
            wrapped_model = EfficientViTSAMForwardWrapper(segmentation_model).eval()

            # EfficientViTSAM expects `img` of shape (B, C, H, W) and `queries` (N, 4)
            # According to the code, image_size is a tuple: self.image_size = (H, W)
            h, w = segmentation_model.image_size
            dummy_img = torch.randn(1, 3, h, w, device=segmentation_model.device)
            # dummy_img = torch.randn(1, 3, 1024, 2048, device=segmentation_model.device)
            dummy_queries = torch.tensor([[0, 0, 50, 50]], device=segmentation_model.device).float()

            model_stats = summary(
                wrapped_model,
                input_data=(dummy_img, dummy_queries),
                col_names=("input_size", "output_size", "num_params", "mult_adds")
            )
        elif model == 'MobileSAM':
            segmentation_model.model.eval()
            wrapped_model = MobileSAMForwardWrapper(segmentation_model).eval()
            # MobileSAM expects image and queries just like EdgeSAM and EfficientViTSAM
            h = w = segmentation_model.image_size  # Assuming image_size is a single integer
            dummy_img = torch.randn(1, 3, h, w, device=segmentation_model.device)
            # dummy_img = torch.randn(1, 3, 1024, 2048, device=segmentation_model.device)
            dummy_queries = torch.tensor([[0, 0, 50, 50]], device=segmentation_model.device).float()
            model_stats = summary(
                wrapped_model,
                input_data=(dummy_img, dummy_queries),
                col_names=("input_size", "output_size", "num_params", "mult_adds")
            )
        elif model == 'SAMHQ':
            segmentation_model.model.eval()
            wrapped_model = SAMHQForwardWrapper(segmentation_model).eval()
            # SAMHQ image_size is likely an int (H=W) like MobileSAM or a tuple. Check the model.
            # Assuming it's a single int or tuple (H,W):
            h = w = segmentation_model.image_size if isinstance(segmentation_model.image_size, int) else segmentation_model.image_size[0]
            dummy_img = torch.randn(1, 3, h, w, device=segmentation_model.device)
            # dummy_img = torch.randn(1, 3, 1024, 2048, device=segmentation_model.device)
            dummy_queries = torch.tensor([[0, 0, 50, 50]], device=segmentation_model.device).float()
            model_stats = summary(
                wrapped_model,
                input_data=(dummy_img, dummy_queries),
                col_names=("input_size", "output_size", "num_params", "mult_adds")
            )
        else:
            print(f'No wrapper for {model} yet.')

        # Extract total params and total mult-adds and store in results dict
        # model_stats is a ModelStatistics object from torchinfo
        results[model][variant] = {
            "TotalParams": model_stats.total_params,
            "TotalMultAdds": model_stats.total_mult_adds
        }

# After processing all models, write results to a JSON file
output_json_path = os.path.join(args.results_path, "model_stats.json")
os.makedirs(args.results_path, exist_ok=True)
with open(output_json_path, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Model stats saved to {output_json_path}")