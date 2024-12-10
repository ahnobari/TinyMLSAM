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

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Function to load data and initialize the model (runs only once)
@st.cache_data
def load_data():
    # Load your data here
    # For example:
    # data = ZeroShotObjectDetectionDataset(path='Data/cityscapes', do_preprocess=False)
    GroundingModel = GDino()
    # Load the dataset
    # data = ZeroShotObjectDetectionDataset(path=args.data_path, prompting=args.use_prompt_engineering)
    data = ZeroShotObjectDetectionDataset(path='Data/cityscapes', do_preprocess=False, processor=GroundingModel.processor)

    batch_size = 8
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)



    boxes = []
    labels = []
    scores = []

    # get the text prompts
    input_ids = data.input_prompt_ins.input_ids.to(GroundingModel.device)
    target_image_size = data.image_size

    for batch in tqdm(dataloader):
        outputs = GroundingModel(batch.to(GroundingModel.device), input_ids=input_ids, target_image_size=target_image_size)
        
        for out in outputs:
            boxes.append(out['boxes'].cpu().numpy())
            labels.append(out['labels'])
            scores.append(out['scores'].cpu().numpy())
            for i in range(len(labels[-1])):
                # only pick first label for any dual predictions
                if labels[-1][i] + '.' not in data.text_prompts:
                    labels[-1][i] = labels[-1][i].split(' ')[0]

   
    return data, boxes

# Function to initialize the model (not cached)
def init_model():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, default='SAM2', help='Model to use for segmentation.')
    argparser.add_argument('--model_variant', type=str, default='facebook/sam2.1-hiera-large', help='Model variant to use for segmentation.')
    argparser.add_argument('--results_path', type=str, default='results', help='Path to save the results.')
    args = argparser.parse_args()

    # Load the model
    SAMModel = getattr(TinySAM, args.model)(args.model_variant)

    # Quantize SAM
    model_quantization(SAMModel, quant_type="nf4")



    return SAMModel

# Function to run the SAM model and plot the results
def run_sam_model(image_index, data, boxes, SAMModel):
    # Load the image from the dataset
    image_np = data.images[image_index]  # Assuming this is a numpy array of shape (1024, 2048, 3)

    # Expand dimensions to add a batch dimension
    image_np_expanded = np.expand_dims(image_np, axis=0)  # Shape becomes (1, 1024, 2048, 3)

    # Run the SAM model
    masks = SAMModel(image_np_expanded, boxes)  # Assuming boxes is defined

    # Convert masks to a format suitable for visualization
    masks_np = masks[0][:5]  # Convert to NumPy array

    # Create a figure to display the image and masks in a grid
    num_masks = len(masks_np)
    num_cols = 3  # Number of columns in the grid
    num_rows = (num_masks // num_cols) + (num_masks % num_cols > 0)  # Calculate number of rows needed

    plt.figure(figsize=(num_cols * 5, num_rows * 5))  # Adjust size based on number of columns and rows

    # Display the original image
    plt.subplot(num_rows, num_cols, 1)
    plt.imshow(image_np)  # Display the original image
    plt.axis('off')

    # Display each mask in the grid
    for i in range(num_masks):
        plt.subplot(num_rows, num_cols, i + 2)  # Start from the second position
        plt.imshow(masks_np[i], alpha=0.5)  # Use alpha for transparency
        plt.axis('off')

    plt.tight_layout()
    st.pyplot(plt)  # Use Streamlit's pyplot function to display the figure

# Load data and boxes
data, boxes = load_data()

# Initialize the model (not cached)
SAMModel = init_model()

# Streamlit app layout
st.title("SAM Model Visualization")
image_index = st.number_input("Enter image number:", min_value=0, max_value=len(data.images)-1, value=0)

if st.button("Run SAM Model"):
    run_sam_model(image_index, data, boxes, SAMModel)

