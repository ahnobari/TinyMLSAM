from TinySAM import *
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm, trange
from torch.utils.data import DataLoader
import argparse
import pickle
import os

argparser = argparse.ArgumentParser()
argparser.add_argument('--model', type=str, default='Base', help='Grounding Dino model to use, Base or Tiny. Default is Base.')
argparser.add_argument('--quantized_variant', type=str, choices=['nf4', 'fp4', 'int8'], default='nf4',
                      help='Quantization variant to use. Options: nf4, fp4, int8')
argparser.add_argument('--data_path', type=str, default='Data/cityscapes', help='Path to the dataset. Default is Data/cityscapes.')
argparser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference. Default is 8.')
argparser.add_argument('--box_threshold', type=float, default=0.2, help='Threshold for bounding box detection. Default is 0.2.')
argparser.add_argument('--text_threshold', type=float, default=0.15, help='Threshold for text detection. Default is 0.15.')
argparser.add_argument('--use_prompt_engineering', action='store_true', help='Use prompt engineering for grounding. Default is False.')
argparser.add_argument('--save_path', type=str, default='results', help='Path to save the instances. Default is results.')
args = argparser.parse_args()

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

datset_folder = args.data_path.split('/')[-1]
pe = 'PE' if args.use_prompt_engineering else 'NoPE'
save_path = os.path.join(args.save_path, f'instances_{args.model}_{datset_folder}_BT_{args.box_threshold}_TT_{args.text_threshold}_{pe}.pkl')

# Load the model
if args.model == 'Base':
    model = GDino(text_threshold=args.text_threshold, box_threshold=args.box_threshold)

else:
    model = GDino("IDEA-Research/grounding-dino-tiny", text_threshold=args.text_threshold, box_threshold=args.box_threshold)

# Quantize model
model_quantization(model, quant_type=args.quantized_variant)

    
# Load the dataset
data = ZeroShotObjectDetectionDataset(path=args.data_path, prompting=args.use_prompt_engineering, processor=model.processor)

loader = DataLoader(data, batch_size=args.batch_size, num_workers=16)

# get the text prompts
input_ids = data.input_prompt_ins.input_ids.to(model.device)
target_image_size = data.image_size

boxes, labels, scores = model.run_loader(loader, input_ids, data.text_prompts, target_image_size)

# Save the instances
final = {'boxes': boxes, 'labels': labels, 'scores': scores}

with open(save_path, 'wb') as f:
    pickle.dump(final, f)