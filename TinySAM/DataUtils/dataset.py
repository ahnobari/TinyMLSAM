from torch.utils.data import Dataset
import torch
import numpy as np
import json
import os
from tqdm.auto import tqdm, trange
from torchvision import transforms
from transformers import AutoProcessor
import gzip
from ..InferenceUtils._vis import show_mask, show_points, show_box
import matplotlib.pyplot as plt

from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.segmentation import MeanIoU

class ZeroShotObjectDetectionDataset(Dataset):
    def __init__(self, path: str, do_preprocess: bool = False, processor: AutoProcessor = None, prompting: bool = True):
        '''
        Dataset for Zero-Shot Object Detection.
        
        Args:
            images (torch.Tensor): Images to be processed.
            do_preprocess (bool): Whether to preprocess the images.
            processor (AutoProcessor): Huggingface AutoProcessor for text processing.
            prompting (bool): Whether to use prompting for the model (must have prompt_dict.json in dataset).
        '''
        
        self.path = path
        
        self.images = np.load(os.path.join(self.path, "images.npy"))
        self.image_size = self.images.shape[1:3]
        
        # load instaces and boxes for scoring
        self.boxes = np.load(os.path.join(self.path, "boxes.npy"))
        self.box_labels = np.load(os.path.join(self.path, "box_labels.npy"))
        
        f = gzip.open(os.path.join(self.path, "instance_masks.npy.gz"), 'rb')
        self.instance_masks = np.load(f)
        f.close()
        
        # pointers for compressed instance_masks saved in a separate file
        self.ptr = np.load(os.path.join(self.path, "ptr.npy"))
        
        self.label_ids = np.load(os.path.join(self.path, "label_ids.npy"))
        
        self.label_dict = json.load(open(os.path.join(self.path, "label_dict.json"), "r"))
        
        if prompting:
            self.prompt_dict = json.load(open(os.path.join(self.path, "prompt_dict.json"), "r"))
        else:
            self.prompt_dict = self.label_dict
        
        self.text_prompts = list(self.prompt_dict.keys())
        for i in range(len(self.text_prompts)):
            self.text_prompts[i] = self.text_prompts[i] + '.'
        
        self.processor = None
        if processor is not None:
            self.input_prompt_ins = processor(text=" ".join(self.text_prompts), return_tensors="pt")
            
            self.processor = processor
        
        self.non_instance_label_ids = []
        self.instance_label_ids = []
        
        self.has_preprocessed = False
        
        if do_preprocess:
            self.preprocess()
        
        # create id map and random colors for each label
        id_map = {}
        id_has_instance = {}
        id_color_map = {}
        for label in self.label_dict:
            id_map[self.label_dict[label]['id']] = label
            id_has_instance[self.label_dict[label]['id']] = self.label_dict[label]['has_instance']
            id_color_map[self.label_dict[label]['id']] = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            
        self.id_map = id_map
        self.id_has_instance = id_has_instance
        self.id_color_map = id_color_map
        
    def __len__(self):
        return len(self.images)
    
    def preprocess(self):
        '''
        Preprocess the dataset.
        '''
        
        self.images_ = torch.zeros(self.images.shape[0], self.images.shape[3], 666, 1333, dtype=torch.float32)
        for i in trange(len(self.images)):
            image = self.__getitem__(i)
            self.images_[i] = image
        self.has_preprocessed = True
    
    def __getitem__(self, idx):
        if self.has_preprocessed:
            return self.images_[idx]
        
        else:
            image = torch.tensor(self.images[idx]).float().permute(2, 0, 1)
            
            if self.processor is not None:
                image = self.processor(images = self.images[idx], return_tensors = "pt").pixel_values[0]
            
            return image
    
    def visualize(self, idx, annotations=True):
        img = self.images[idx]
        masks = self.instance_masks[self.ptr[idx]:self.ptr[idx+1]]
        boxes = self.boxes[self.ptr[idx]:self.ptr[idx+1]]
        box_labels = self.box_labels[self.ptr[idx]:self.ptr[idx+1]]
        
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        
        for i, mask in enumerate(masks):
            show_mask(mask, plt.gca(), color=self.id_color_map[box_labels[i]])
            show_box(boxes[i], plt.gca(), color=self.id_color_map[box_labels[i]])
            if annotations:
                # add annotation
                plt.text(boxes[i][0], boxes[i][1], self.id_map[box_labels[i]], size=10, color='black', ha = 'left', va = 'bottom', bbox=dict(boxstyle="square", facecolor=self.id_color_map[box_labels[i]], edgecolor=self.id_color_map[box_labels[i]], alpha=1.0, pad=0.0))
            
    def visualize_prediction(self, idx, boxes, masks, labels, unify=False, annotations=True):
        img = self.images[idx]
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')

        if not unify:
            for i, mask in enumerate(masks):
                if isinstance(labels[i], int):
                    id_ = labels[i]
                else:
                    id_ = self.prompt_dict[labels[i]]['id']
                show_mask(mask, plt.gca(), color=self.id_color_map[id_])
                show_box(boxes[i], plt.gca(), color=self.id_color_map[id_])
                if annotations:
                    plt.text(boxes[i][0], boxes[i][1], self.id_map[id_], size=10, color='black', ha = 'left', va = 'bottom', bbox=dict(boxstyle="square", facecolor=self.id_color_map[id_], edgecolor=self.id_color_map[id_], alpha=1.0, pad=0.0))
        else:
            final_boxes = []
            final_masks = []
            final_labels = []
            unified_mask = np.zeros([self.image_size[0], self.image_size[1]], dtype=np.int64)
            
            for j in range(len(boxes)):
                if isinstance(labels[j], int):
                    id_ = labels[j]
                else:
                    id_ = self.prompt_dict[labels[j]]['id']
                if self.id_has_instance[id_]:
                    final_boxes.append(boxes[j])
                    final_masks.append(masks[j])
                    final_labels.append(id_)
                else:
                    if id_ not in final_labels:
                        final_labels.append(id_)
                        final_masks.append(masks[j])
                        final_boxes.append(boxes[j])
                    else:
                        ind = final_labels.index(id_)
                        final_masks[ind] = np.logical_or(final_masks[ind], masks[j])
                        x,y = np.where(final_masks[ind])
                        final_boxes[ind] = np.array([np.min(y), np.min(x), np.max(y), np.max(x)])

                unified_mask[np.logical_and(unified_mask == 0 , masks[j])] = id_
            
            for i in range(len(final_boxes)):
                show_mask(final_masks[i], plt.gca(), color=self.id_color_map[final_labels[i]])
                show_box(final_boxes[i], plt.gca(), color=self.id_color_map[final_labels[i]])
                if annotations:
                    plt.text(final_boxes[i][0], final_boxes[i][1], self.id_map[final_labels[i]], size=10, color='black', ha = 'left', va = 'bottom', bbox=dict(boxstyle="square", facecolor=self.id_color_map[final_labels[i]], edgecolor=self.id_color_map[final_labels[i]], alpha=1.0, pad=0.0))
            
            # finally plot the unified masks next to each other
            fig, axs = plt.subplots(1, 2, figsize=(20, 10))
            unique_labels = np.unique(unified_mask)
            for label in unique_labels:
                if label == 0:
                    continue
                mask = unified_mask == label
                show_mask(mask, axs[0], color=self.id_color_map[label])

            axs[0].axis('off')
            axs[0].set_title("Unified Mask Prediction")
            
            unique_labels = np.unique(self.label_ids[idx])
            for label in unique_labels:
                if label == 0:
                    continue
                mask = self.label_ids[idx] == label
                show_mask(mask, axs[1], color=self.id_color_map[label])
                
            axs[1].axis('off')
            axs[1].set_title("Ground Truth Mask")
            
    
    def evaluate_precitions(self, boxes, labels, masks, scores, return_processed=False, per_class_stats=True):
        # Scores are expected to be sorted in descending order
        
        if len(boxes) != self.__len__() or len(labels) != self.__len__() or len(masks) != self.__len__() or len(scores) != self.__len__():
            raise ValueError("Length of predictions should be equal to the length of the dataset.")

        iou = MeanIoU(num_classes = len(self.label_dict)+1, include_background=False, input_format='index', per_class=per_class_stats)
        map = MeanAveragePrecision(iou_type='segm', class_metrics=per_class_stats)
        overall_iou = 0
        
        if return_processed:
            processed_boxes = []
            processed_masks = []
            processed_labels = []
            processed_scores = []
            unified_masks = []
        
        for i in trange(len(boxes)):
            # first process non-instance labels into a single box and mask
            final_boxes = []
            final_masks = []
            final_labels = []
            final_scores = []
            unified_mask = np.zeros([self.image_size[0], self.image_size[1]], dtype=np.int64)
            
            for j in range(len(boxes[i])):
                if isinstance(labels[i][j], int):
                    id_ = labels[i][j]
                else:
                    id_ = self.prompt_dict[labels[i][j]]['id']
                if self.id_has_instance[id_]:
                    final_boxes.append(boxes[i][j])
                    final_masks.append(masks[i][j])
                    final_labels.append(id_)
                    final_scores.append(scores[i][j])
                else:
                    if id_ not in final_labels:
                        final_labels.append(id_)
                        final_scores.append(scores[i][j])
                        final_masks.append(masks[i][j])
                        final_boxes.append(boxes[i][j])
                    else:
                        idx = final_labels.index(id_)
                        final_masks[idx] = np.logical_or(final_masks[idx], masks[i][j])
                        x,y = np.where(final_masks[idx])
                        final_boxes[idx] = np.array([np.min(y), np.min(x), np.max(y), np.max(x)])

                unified_mask[np.logical_and(unified_mask == 0 ,masks[i][j])] = id_
            
            if return_processed:
                processed_boxes.append(final_boxes)
                processed_masks.append(final_masks)
                processed_labels.append(final_labels)
                processed_scores.append(final_scores)
                unified_masks.append(unified_mask)
            
            final_boxes = torch.tensor(np.array(final_boxes)).float()
            final_masks = torch.tensor(np.array(final_masks)).bool()
            final_labels = torch.tensor(np.array(final_labels)).long()
            final_scores = torch.tensor(np.array(final_scores)).float()

            intersection = (unified_mask == self.label_ids[i])[np.logical_and(unified_mask != 0, self.label_ids[i] != 0)].sum()
            union = ((unified_mask + self.label_ids[i]) != 0).sum()
            overall_iou += intersection/union/len(boxes)
            
            unified_mask = torch.tensor(unified_mask).long()
            
            iou.update(unified_mask, torch.tensor(self.label_ids[i]).long())
            
            preds = [{
                'boxes': final_boxes,
                'labels': final_labels,
                'scores': final_scores,
                'masks': final_masks
            }]
            
            targets = [{
                'boxes': torch.tensor(self.boxes[self.ptr[i]:self.ptr[i+1]]).float(),
                'labels': torch.tensor(self.box_labels[self.ptr[i]:self.ptr[i+1]]).long(),
                'masks': torch.tensor(self.instance_masks[self.ptr[i]:self.ptr[i+1]]).bool()
            }]
            
            map.update(preds, targets)
        
        mIoU = iou.compute()
        mAP = map.compute()
        print("Mean IoU: ", mIoU)
        print("Average Whole Image IoU: ", overall_iou)
        print("Precision Scores: ", mAP)
        
        if return_processed:
            return mIoU, mAP, overall_iou, processed_boxes, processed_labels, processed_masks, processed_scores, unified_masks
        else:
            return mIoU, mAP, overall_iou
        
            
        