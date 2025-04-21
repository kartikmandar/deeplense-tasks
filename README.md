# DeepLense: Gravitational Lensing Analysis

This repository contains implementations of deep learning models for the analysis of gravitational lensing images, completed as part of the DeepLense tasks. The repository is organized into three main components:

## Repository Structure

```
/
├── task1/               # Original multi-class classification implementation
├── task2/               # Original binary lens finder implementation
└── updated/             # Enhanced implementations with multiple model architectures
    └── task1/           # Comprehensive model comparison for task1
```

## Task 1: Multi-Class Gravitational Lens Substructure Classification

### Original Implementation (`task1/`)

The original implementation in the `task1/` directory focuses on a single high-performance model (ConvNeXt V2 Tiny) for classifying gravitational lensing images into three categories:
- **No Substructure (`no`)**: Images without noticeable substructure
- **Sphere Substructure (`sphere`)**: Images with spherical substructures similar to cold dark matter subhalos
- **Vortex Substructure (`vort`)**: Images with vortex-like patterns

**Key Features:**
- Utilizes `facebook/convnextv2-tiny-1k-224` with ImageNet pre-trained weights
- Achieves 0.9943 Test ROC AUC (OVR Macro) and 0.9544 Test Accuracy
- Includes visualizations for training history, confusion matrix, and ROC curve

[See full details in task1/README.md](task1/README.md)

### Enhanced Implementation (`updated/task1/`)

The enhanced implementation in `updated/task1/` provides a comprehensive comparison of multiple model architectures for the same classification task. This implementation:

- Evaluates 13 different model architectures:
  - ResNet family (18, 34, 50, 101, 152)
  - EfficientNet family (B0, B1, B2, B3, B4, B5)
  - MobileNetV3 (Small, Large)

- Provides extensive performance comparisons:
  - Test AUC and accuracy metrics for all models
  - Training time comparisons
  - AUC vs. training time analysis

- Includes a complete pipeline for:
  - Training individual models
  - Generating performance visualizations
  - Comparing model results

**Key Findings:**
- ResNet34 achieved the highest Test AUC (0.9942), closely followed by EfficientNet-B1 (0.9937)
- MobileNetV3 models trained the fastest but with slightly lower AUC scores
- Comprehensive visualizations for all models including confusion matrices, ROC curves, and training histories

[See full details in updated/task1/README.md](updated/task1/README.md)

## Task 2: Gravitational Lens Finding (Binary Classification)

The implementation in `task2/` focuses on binary classification to distinguish images containing strong gravitational lenses from those without, addressing significant class imbalance (1:16 in training, 1:100 in testing).

**Key Features:**
- Uses `facebook/convnextv2-tiny-1k-224` with weighted loss to address class imbalance
- Achieves 0.9925 Test ROC AUC and 0.9810 Test Accuracy
- High recall (0.92) for the lens class ensures most actual lenses are detected
- Includes visualizations of training history, confusion matrix, and ROC curve

[See full details in task2/README.md](task2/README.md)

## Model Weights

Pre-trained model weights are available for download:

- **Task 1 (Original Implementation):**
  [Download Task1 Model](https://drive.google.com/file/d/1uaLVsV-xLv7fDwJmlwU8eBCe0f0WILwV/view?usp=sharing)

- **Task 2 (Binary Classification):**
  [Download Task2 Model](https://drive.google.com/file/d/14U7lbo39qMqPoPSLAyHy_jGiEHa_3BVN/view?usp=sharing)

- **Updated Task 1 (Multiple Models):**
  [Download All Updated Models](https://drive.google.com/drive/folders/1DwOVTB2IjXUZTEbIBWvxiRRdx2t_KtEe)

## Requirements

The project requires the following dependencies:

- Python 3.6+
- PyTorch and torchvision
- NumPy
- Matplotlib
- Pandas
- Seaborn
- scikit-learn
- tqdm
- Hugging Face Transformers
- timm (for additional model architectures in updated implementation)

## Acknowledgements

This project is part of the DeepLense initiative, which aims to apply deep learning techniques to gravitational lensing analysis for advancing our understanding of dark matter and the universe.