# MRNet Multi-class Knee MRI Classification Notebook

## Overview
This notebook trains and evaluates a **multi-class classifier** for knee MRI abnormality detection using the **MRNet** dataset on **Kaggle**.

The notebook uses:
- **Sagittal plane only**
- **Exclusive 4-class labeling**
- **Transfer learning**
- Comparison-ready backbones such as:
  - **ResNet50**
  - **InceptionV3**
  - **ViT-small**
  - **ViT-large**

## Task Definition
This notebook uses a **multi-class** setup, meaning each MRI exam is assigned to exactly **one** class.

Because MRNet labels overlap in the raw dataset, the notebook converts them into exclusive classes using the following priority rule:

1. **ACL Tear**
2. **Meniscus Tear**
3. **Abnormal Other**
4. **Normal**

### Class Mapping
- `0 = ACL Tear`
- `1 = Meniscus Tear`
- `2 = Abnormal Other`
- `3 = Normal`

## Dataset
Root path used in Kaggle:

`/kaggle/input/datasets/cjinny/mrnet-v1/MRNet-v1.0`

Expected structure:
- `train/sagittal/*.npy`
- `valid/sagittal/*.npy`
- `train-acl.csv`
- `train-meniscus.csv`
- `train-abnormal.csv`
- `valid-acl.csv`
- `valid-meniscus.csv`
- `valid-abnormal.csv`

Each `.npy` file contains a knee MRI volume with shape roughly:

`[num_slices, H, W]`

## Main Pipeline
The notebook performs the following steps:

1. Load MRNet labels from CSV files
2. Build exclusive multi-class targets
3. Load sagittal MRI volumes from `.npy`
4. Select informative slices from each exam
5. Apply preprocessing and augmentation
6. Encode slices using a pretrained backbone
7. Aggregate slice features at the exam level
8. Predict one of the 4 classes
9. Evaluate using multiple metrics

## Preprocessing
The notebook includes:
- conversion of grayscale slices to RGB format
- resizing to `224 x 224`
- normalization
- data augmentation such as:
  - random rotation
  - random translation
  - horizontal flip

## Models
The notebook supports these model types:
- `resnet50`
- `inception_v3`
- `vit_small`
- `vit_large`

### ViT Design
For ViT models:
- a pretrained **timm** ViT image encoder is used
- slice embeddings are extracted
- a transformer-based sequence encoder aggregates slice information
- a classification head outputs the final class

## Loss and Optimization
The notebook supports:
- **class-weighted cross-entropy**
- optional **focal loss**

It also uses:
- separate learning rates for backbone and classification head
- cosine annealing learning rate scheduling

## Metrics Reported
The notebook reports:
- **Macro AUC (one-vs-rest)**
- **Accuracy**
- **Balanced Accuracy**
- **Macro F1-score**
- **Confusion Matrix**
- **Classification Report**

These metrics are important because the class distribution is imbalanced.

## Output Files
The notebook saves:
- model weights
- prediction CSV
- summary metrics

Typical output paths:
- `/kaggle/working/final_multiclass_<model>_sagittal.pth`
- `/kaggle/working/preds_multiclass_<model>_sagittal.csv`

## How to Run
1. Open the notebook in Kaggle
2. Make sure the MRNet dataset is attached
3. Set the model type in the config cell:
   - `resnet50`
   - `inception_v3`
   - `vit_small`
   - `vit_large`
4. Run all cells in order
5. Review the final metrics and saved prediction files

## Recommended Experiments
Suggested comparisons:
- ResNet50 vs InceptionV3
- ResNet50 vs ViT-small
- ViT-small vs ViT-large
- cross-entropy vs focal loss

## Important Limitation
This notebook uses a **multi-class approximation** of MRNet.  
In the original dataset, labels can overlap, so this formulation is less natural than a **multi-label** setup.

That means:
- results should be interpreted as a useful experimental baseline
- the notebook is best used for comparison against more realistic multi-label approaches

## Best Use Case
Use this notebook when you want:
- a clean **multi-class baseline**
- a coursework-friendly comparison model
- a direct evaluation of different pretrained backbones on MRNet

## Author Notes
This notebook is suitable for:
- coursework experimentation
- model comparison
- reporting baseline multi-class results

For stronger alignment with MRNet label structure, a **multi-label notebook** is usually preferred.
