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

argparser = argparse.ArgumentParser()
argparser.add_argument('--model', type=str, default='SAM2', help='Model to use for segementation. Default is SAM2. Options: SAM2, EdgeSAM, EfficientSAM, EfficientViTSAM, MobileSAM, SAMHQ')
argparser.add_argument('--model_variant', type=str, default='facebook/sam2.1-hiera-large', help='Model variant to use for segmentation. Default is facebook/sam2.1-hiera-large. Will depend on the model chosen.')
argparser.add_argument('--quantized_variant', type=str, choices=['nf4', 'fp4', 'int8'], default='nf4',
                      help='Quantization variant to use. Options: nf4, fp4, int8')
argparser.add_argument('--data_path', type=str, default='Data/cityscapes', help='Path to the dataset. Default is Data/cityscapes.')
argparser.add_argument('--queries_path', type=str, default='results/instances_Base_cityscapes_BT_0.2_TT_0.15_PE.pkl', help='Path to the grounding instances. Default is results/instances_Base_cityscapes_BT_0.2_TT_0.15.pkl.')
argparser.add_argument('--save_masks', action='store_true', help='Save the segmentation masks. Default is False.')
argparser.add_argument('--results_path', type=str, default='results', help='path to save the results (if save_masks is passed the raw results will also be saved). Default is results. File name will include run information and a unique id.')
argparser.add_argument('--use_prompt_engineering', action='store_true', help='Use prompt engineering for grounding. Default is False.')
args = argparser.parse_args()

# check if the results path exists
if not os.path.exists(args.results_path):
    os.makedirs(args.results_path)
    
# create a unique id for the run
while True:
    uid = str(uuid.uuid4())
    model_safe = args.model_variant.replace("/", "_")
    f_name = os.path.join(args.results_path, f"{args.model}_{model_safe}_{uid}.pkl")
    if not os.path.exists(f_name):
        break

# make an empty results dictionary
if args.save_masks:
    results = {"mIoU": None, "mAP": None, "overall_iou": None, "processed_boxes": None, "processed_labels": None, "processed_masks": None, "processed_scores": None, "unified_masks": None}
else:
    results = {"mIoU": None, "mAP": None, "overall_iou": None}

# save the empty results dictionary
with open(f_name, 'wb') as f:
    pickle.dump(results, f)
    
# Load the model
SAMModel = getattr(TinySAM, args.model)(args.model_variant)

# Quantize SAM
model_quantization(SAMModel, quant_type=args.quantized_variant)

    
# Load the dataset
data = ZeroShotObjectDetectionDataset(path=args.data_path, prompting=args.use_prompt_engineering)

# get annotations
with open(args.queries_path, 'rb') as f:
    queries = pickle.load(f)
    
boxes = queries['boxes']
labels = queries['labels']
scores = queries['scores']

# Run the SAM model
masks = SAMModel(data.images, boxes)

# Evaluate the results
if args.save_masks:
    mIoU, mAP, overall_iou, processed_boxes, processed_labels, processed_masks, processed_scores, unified_masks = data.evaluate_precitions(boxes, labels, masks, scores, return_processed=True)
    results["processed_boxes"] = processed_boxes
    results["processed_labels"] = processed_labels
    results["processed_masks"] = processed_masks
    results["processed_scores"] = processed_scores
    results["unified_masks"] = unified_masks
    results["mIoU"] = mIoU
    results["mAP"] = mAP
    results["overall_iou"] = overall_iou
else:
    mIoU, mAP, overall_iou = data.evaluate_precitions(boxes, labels, masks, scores, return_processed=False)
    results["mIoU"] = mIoU
    results["mAP"] = mAP
    results["overall_iou"] = overall_iou
    
# save the results
with open(f_name, 'wb') as f:
    pickle.dump(results, f)