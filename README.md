# README

## Model Variants
<table border="1" cellpadding="8" cellspacing="0">
  <thead>
    <tr>
      <th>Grounding Dino Variant</th>
      <th>SAM Models</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="18">Base</td>
      <td>EdgeSAM</td>
    </tr>
    <tr><td>EdgeSAM_3x</td></tr>
    <tr><td>EfficientSAM_S</td></tr>
    <tr><td>EfficientSAM_Ti</td></tr>
    <tr><td>EfficientViTSAM_xl1</td></tr>
    <tr><td>EfficientViTSAM_xl0</td></tr>
    <tr><td>EfficientViTSAM_l2</td></tr>
    <tr><td>EfficientViTSAM_l1</td></tr>
    <tr><td>EfficientViTSAM_l0</td></tr>
    <tr><td>MobileSAM</td></tr>
    <tr><td>facebook/sam2.1-hiera-tiny</td></tr>
    <tr><td>facebook/sam2.1-hiera-small</td></tr>
    <tr><td>facebook/sam2.1-hiera-large</td></tr>
    <tr><td>facebook/sam2.1-hiera-base-plus</td></tr>
    <tr><td>SAMHQ_vit_h</td></tr>
    <tr><td>SAMHQ_vit_l</td></tr>
    <tr><td>SAMHQ_vit_b</td></tr>
    <tr><td>SAMHQ_vit_tiny</td></tr>
    <tr>
      <td rowspan="18">Tiny</td>
      <td>EdgeSAM</td>
    </tr>
    <tr><td>EdgeSAM_3x</td></tr>
    <tr><td>EfficientSAM_S</td></tr>
    <tr><td>EfficientSAM_Ti</td></tr>
    <tr><td>EfficientViTSAM_xl1</td></tr>
    <tr><td>EfficientViTSAM_xl0</td></tr>
    <tr><td>EfficientViTSAM_l2</td></tr>
    <tr><td>EfficientViTSAM_l1</td></tr>
    <tr><td>EfficientViTSAM_l0</td></tr>
    <tr><td>MobileSAM</td></tr>
    <tr><td>facebook/sam2.1-hiera-tiny</td></tr>
    <tr><td>facebook/sam2.1-hiera-small</td></tr>
    <tr><td>facebook/sam2.1-hiera-large</td></tr>
    <tr><td>facebook/sam2.1-hiera-base-plus</td></tr>
    <tr><td>SAMHQ_vit_h</td></tr>
    <tr><td>SAMHQ_vit_l</td></tr>
    <tr><td>SAMHQ_vit_b</td></tr>
    <tr><td>SAMHQ_vit_tiny</td></tr>
  </tbody>
</table>



## Setting up environment
### Installation steps using conda 
1. Install conda using the conda installation instructions from https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
2. Create a conda environment using the `environment.yml` file
3. Activate the environment `conda activate TinySAM`
3. Setup the latest version of pytorch and other dependencies using `conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia`
4. Install mmdet using `conda install mmdet`
5. Install lower version of pytorch for mmcv 2.1 `conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 mmcv==2.1.0 -c pytorch -c nvidia`. 

### Installation steps using mamba 
1. Install mamba using the installation instructions from https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html
2. Create a mamba environment using the `environment.yml` file 
3. Activate the environment `mamba activate TinySAM`
3. Setup the latest version of pytorch and other dependencies using `mamba install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia`
4. Install mmdet using `mamba install mmdet` 
5. Install lower version of pytorch for mmcv 2.1 `mamba install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 mmcv==2.1.0 -c pytorch -c nvidia`. 

###  Download the dataset
We evaluate the SAM variants on the following two datasets: [cityscapes](https://www.cityscapes-dataset.com/) and [BDD100K](https://www.vis.xyz/bdd100k/). Our choice of these datasets depends on our usecase of benchmarking different efficient variants of SAM on autonomous driving datasets under different accuracy, memory, and compute constraints. 

- To download the cityscapes dataset, navigate to ~/Data/cityscapes and execute `python runthis.py`

- To download the BDD100K dataset, navigate to ~/Data/BDD100K and execute `python runthis.py`

### Download the pretrained checkpoints 
To download the pretrained checkpoints, navigate inside ~/weights and execute `python runthis.py`

## Usage
### We provide a total of 18 SAM variants 
| Model            | Variants                                                                                   |
|------------------|--------------------------------------------------------------------------------------------|
| SAM2             | facebook/sam2.1-hiera-tiny, facebook/sam2.1-hiera-small, facebook/sam2.1-hiera-large, facebook/sam2.1-hiera-base-plus |
| EdgeSAM          | Base, _3x                                                                                  |
| EfficientSAM     | S, Ti                                                                                      |
| EfficientViTSAM  | efficientvit-sam-xl1, efficientvit-sam-l0, efficientvit-sam-l1, efficientvit-sam-l2, efficientvit-sam-xl0 |
| MobileSAM        | MobileSAM                                                                                  |
| SAMHQ            | vit_h, vit_l, vit_b, vit_tiny                                                              |

- To compute the metrics for all variants on the cityscapes and BDD100K datasets, execute the bash script `./run_all_sams.sh`.  
- To compute the metrics for a specific model and variant on either cityscapes or BDD100K dataset, execute the following script 
```bash
# Example usage:
python run_segmentation.py \
    --model SAM2 \
    --model_variant facebook/sam2.1-hiera-large \
    --data_path Data/cityscapes \
    --queries_path results/instances_Base_cityscapes_BT_0.2_TT_0.15_PE.pkl \
    --save_masks \
    --results_path results \
    --use_prompt_engineering

# This command:
# - Uses the SAM2 model with the 'facebook/sam2.1-hiera-large' variant
# - Processes data from 'Data/cityscapes'
# - Loads queries from 'results/instances_Base_cityscapes_BT_0.2_TT_0.15_PE.pkl'
# - Saves the segmentation masks to the results directory
# - Enables prompt engineering for grounding (instance should have been predicted with prompt engineering)
```

Before running the above you will need to first compute queries using Grounding Dino. To do so run `./run_all_dino.sh`
You can also specify Dino model settings by running the following script:
```bash
#Example usage:
python get_instances.py \
    --model Base \
    --data_path Data/cityscapes \
    --batch_size 8 \
    --box_threshold 0.2\
    --text_threshold 0.15 \
    --save_path results \
    --use_prompt_engineering

# This command:
# - Uses the Base variant of grounding dino model (can be Tiny)
# - Processes data from 'Data/cityscapes'
# - use a batch size of 8
# - uses a box threshold of 0.2
# - uses a text threshold of 0.15
# - Saves the instances predicted in the results folder
# - Enables prompt engineering for grounding
```

<!-- Very shady but works! -->
