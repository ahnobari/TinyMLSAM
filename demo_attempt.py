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



@st.cache_data
def init():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, default='SAM2', help='Model to use for segementation. Default is SAM2. Options: SAM2, EdgeSAM, EfficientSAM, EfficientViTSAM, MobileSAM, SAMHQ')
    argparser.add_argument('--model_variant', type=str, default='facebook/sam2.1-hiera-large', help='Model variant to use for segmentation. Default is facebook/sam2.1-hiera-large. Will depend on the model chosen.')
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
        Dino_safe = args.queries_path.split("/")[-1].replace(".pkl", "")
        f_name = os.path.join(args.results_path, f"{args.model}_{model_safe}_{Dino_safe}_{uid}.pkl")
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
        
    GroundingModel = GDino()
    # Load the dataset
    # data = ZeroShotObjectDetectionDataset(path=args.data_path, prompting=args.use_prompt_engineering)
    data = ZeroShotObjectDetectionDataset(path=args.data_path, do_preprocess=False, processor=GroundingModel.processor)

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

    return data, boxes, SAMModel

# np.expand_dims(image_np, axis=0)
# print(np.expand_dims(data.images[0], axis=0).shape, len(boxes))

# Assuming SAMModel and data are defined elsewhere in your code
# from your_module import SAMModel, data

# Function to run the SAM model and plot the results
def run_sam_model(image_index):
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


data, boxes, SAMModel = init()

# Streamlit app layout
st.title("SAM Model Visualization")
image_index = st.number_input("Enter image number:", min_value=0, max_value=len(data.images)-1, value=0)

if st.button("Run SAM Model"):
    run_sam_model(image_index)


"""
# Run the SAM model
masks = SAMModel(np.expand_dims(data.images[0], axis=0), boxes[0])


# Load the first image from the dataset
image_np = data.images[0]  # Assuming this is a numpy array of shape (1024, 2048, 3)

# Expand dimensions to add a batch dimension
image_np_expanded = np.expand_dims(image_np, axis=0)  # Shape becomes (1, 1024, 2048, 3)

# Run the SAM model
masks = SAMModel(image_np_expanded, boxes)  # Assuming boxes[0] is the correct format

# print(masks)

# Convert masks to a format suitable for visualization
# Assuming masks is a tensor of shape (N, H, W) where N is the number of masks
masks_np = masks[0][:5] # .detach().cpu().numpy()  # Convert to NumPy array

# Create a figure to display the image and masks in a grid
num_masks = len(masks_np)
num_cols = 3  # Number of columns in the grid
num_rows = (num_masks // num_cols) + (num_masks % num_cols > 0)  # Calculate number of rows needed

plt.figure(figsize=(num_cols * 5, num_rows * 5))  # Adjust size based on number of columns and rows

# Display the original image
plt.subplot(num_rows, num_cols, 1)
plt.imshow(image_np)  # Display the original image
# plt.title("Original Image")
plt.axis('off')

# Display each mask in the grid
for i in range(num_masks):
    plt.subplot(num_rows, num_cols, i + 2)  # Start from the second position
    plt.imshow(masks_np[i], alpha=0.5)  # Use alpha for transparency
    # plt.title(f"Mask {i + 1}")
    plt.axis('off')

plt.tight_layout()
plt.show()
"""
"""
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
"""
