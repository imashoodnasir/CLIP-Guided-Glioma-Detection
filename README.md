
# Explainable CLIP-Guided Hybrid Vision Transformer (ECH-ViT)

## Overview
Implementation of:
Explainable CLIP-Guided Hybrid Vision Transformer for Multi-Center Glioma Segmentation

## Features
- Hybrid CNN + Transformer backbone
- CLIP-based semantic guidance
- Cross-attention fusion
- Multi-class 3D segmentation
- Explainability via attention maps

## Installation
pip install torch torchvision monai timm transformers nibabel

## Training
python train.py

## Structure
- dataset.py
- models/ech_vit.py
- train.py
- evaluate.py

## Notes
Fill dataset case paths inside train.py before training.
