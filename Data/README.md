## Basic Instance Segmentation Test Dataset Format

### Images: (file: images.npy, shape: (H,W,C))
All of the test images in their native resolution

### Label Dictionary: (file: label_dict.json)
A dictionary with the following format in json:

```json
{
    "ClassLabel_0": {
        "id": 1,
        "has_instance": true/false
    },
    "ClassLabel_1": {
        "id": 2,
        "has_instance": true/false
    },
    ...
}
```

### Labeled Masks (file: label_ids.npy, shape: (H,W))
A set of single channel images with integer index of the corresponding classes for the mask portions.

### Boxes (file: boxes.npy, shape (M,4))
The boxes for all instances in each image in one array (seperated by ptr)

### Box Labels (file box_labels.npy, shape (N))
Integer index of the labels for each box.

### Insatnce Masks - Gzip Compression level 2 (file: instance_masks.npy.gz, shape: (M,H,W))
The binary mask for each box instance.

### Pointer (file: ptr.npy, shape: (N+1,))
Pointer to start and end of images instances. Similar to a CSR sparse matrix.
