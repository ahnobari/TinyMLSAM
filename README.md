# To Be Written!!

## Model Variants
<table>
  <thead>
    <tr>
      <td>Grounding Dino Variant</td>
      <td>SAM Model</td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan=18>Base</td>
      <td>EdgeSAM</td>
    </tr>
    <tr>
      <td>EdgeSAM_3x</td>
    </tr>
    <tr>
      <td>EfficientSAM_S</td>
    </tr>
    <tr>
      <td>EfficientSAM_Ti</td>
    </tr>
    <tr>
      <td>EfficientViTSAM_xl1</td>
    </tr>
    <tr>
      <td>EfficientViTSAM_xl0</td>
    </tr>
    <tr>
      <td>EfficientViTSAM_l2</td>
    </tr>
    <tr>
      <td>EfficientViTSAM_l1</td>
    </tr>
    <tr>
      <td>EfficientViTSAM_l0</td>
    </tr>
    <tr>
      <td>MobileSAM</td>
    </tr>
    <tr>
      <td>facebook/sam2.1-hiera-tiny</td>
    </tr>
    <tr>
      <td>facebook/sam2.1-hiera-small</td>
    </tr>
    <tr>
      <td>facebook/sam2.1-hiera-large</td>
    </tr>
    <tr>
      <td>facebook/sam2.1-hiera-base-plus</td>
    </tr>
    <tr>
      <td>SAMHQ_vit_h</td>
    </tr>
    <tr>
      <td>SAMHQ_vit_l</td>
    </tr>
    <tr>
      <td>SAMHQ_vit_b</td>
    </tr>
    <tr>
      <td>SAMHQ_vit_tiny</td>
    </tr>
    <tr>
      <td rowspan=18>Tiny</td>
      <td>EdgeSAM</td>
    </tr>
    <tr>
      <td>EdgeSAM_3x</td>
    </tr>
    <tr>
      <td>EfficientSAM_S</td>
    </tr>
    <tr>
      <td>EfficientSAM_Ti</td>
    </tr>
    <tr>
      <td>EfficientViTSAM_xl1</td>
    </tr>
    <tr>
      <td>EfficientViTSAM_xl0</td>
    </tr>
    <tr>
      <td>EfficientViTSAM_l2</td>
    </tr>
    <tr>
      <td>EfficientViTSAM_l1</td>
    </tr>
    <tr>
      <td>EfficientViTSAM_l0</td>
    </tr>
    <tr>
      <td>MobileSAM</td>
    </tr>
    <tr>
      <td>facebook/sam2.1-hiera-tiny</td>
    </tr>
    <tr>
      <td>facebook/sam2.1-hiera-small</td>
    </tr>
    <tr>
      <td>facebook/sam2.1-hiera-large</td>
    </tr>
    <tr>
      <td>facebook/sam2.1-hiera-base-plus</td>
    </tr>
    <tr>
      <td>SAMHQ_vit_h</td>
    </tr>
    <tr>
      <td>SAMHQ_vit_l</td>
    </tr>
    <tr>
      <td>SAMHQ_vit_b</td>
    </tr>
    <tr>
      <td>SAMHQ_vit_tiny</td>
    </tr>
  </tbody>
</table>


## Setting up environment

1. First use the `environmnet.yml` to make a conda/mamba environment.
2. run the install command for current version of pytorch: `mamba install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia`
3. install mmdet `mamba install mmdet`
4. install lower version of pytorch with mmcv 2.1 `mamba install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 mmcv==2.1.0 -c pytorch -c nvidia`

Very shady but works!