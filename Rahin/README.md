# Rahin - MRNet Knee MRI Classification

This folder contains my MRNet knee MRI classification notebook for the Applied Machine Learning coursework.

## Notebook

The notebook implements a multi-label, multi-view Vision Transformer model for predicting:

- ACL tear
- Meniscus tear
- Abnormality

## Dataset

Expected Kaggle dataset path:

`/kaggle/input/datasets/cjinny/mrnet-v1/MRNet-v1.0`

## Main features

- Dataset exploration and visualisation
- Multi-label classification
- All three MRI planes: sagittal, coronal, axial
- Informative slice selection
- Shared pretrained ViT encoder from `timm`
- Gated multi-view fusion
- Separate heads for ACL, meniscus, and abnormal
- Focal loss
- Per-label threshold tuning
- AUC, balanced accuracy, precision, recall, F1, and macro F1
- External image inference section
