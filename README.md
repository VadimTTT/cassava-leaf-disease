# Cassava Leaf Disease Classification

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue)](https://www.kaggle.com/c/cassava-leaf-disease-classification)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)](https://pytorch.org/)

> **Final Private Score: 0.8741 (87.41%)**  
> **Public Score: 0.8733 (87.33%)**  

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Challenge & Approach](#challenge--approach)
- [Methodology](#methodology)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation & Usage](#installation--usage)
- [Key Learnings](#key-learnings)
- [Future Improvements](#future-improvements)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project tackles the **Cassava Leaf Disease Classification** challenge from Kaggle. The goal is to classify cassava leaf images into **5 disease categories** (including healthy) using computer vision techniques.

The dataset is highly imbalanced, with the smallest class representing only **5.08%** of the data. To achieve robust performance, I implemented:

*   **5-fold Stratified Cross-Validation**
*   **Transfer Learning** with EfficientNet-B1
*   **Data Augmentation** to improve generalization
*   **Model Ensemble** (5 models) for final predictions

The final ensemble achieved **87.41% accuracy** on the hidden test set.

---

## Dataset

| Split | Images | Classes | Imbalance |
|:--- | :--- | :--- | :--- |
| Train | 21,397 | 5 | Class 0: 61.5%, Class 4: 5.1% |
| Test (public) | 1 | 5 | Hidden (≈15,000 images) |

### Class Distribution
* **Class 0 (CMD):** 13,158 images <progress value="61.5" max="100"></progress> 61.5%
* **Class 1 (CBB):** 2,577 images <progress value="12.0" max="100"></progress> 12.0%
* **Class 2 (CGM):** 2,386 images <progress value="11.2" max="100"></progress> 11.2%
* **Class 3 (CBSD):** 2,189 images <progress value="10.2" max="100"></progress> 10.2%
* **Class 4 (Healthy):** 1,087 images <progress value="5.1" max="100"></progress> 5.1%

### Challenge Features
*   **Class imbalance** – dominant class (CMD) vs minority (Healthy).
*   **Real-world images** – varying lighting, angles, and leaf positions.
*   **Hidden test set** – the actual test data is only revealed during submission.
*   **Internet disabled** – models must be packaged locally for Kaggle submission.

---

## Challenge & Approach

### Problem 1: Class Imbalance
**Solution:** Stratified K-Fold ensures each fold maintains class distribution. `CrossEntropyLoss` with class weights was used to penalize minority class mistakes more heavily.

### Problem 2: Limited Training Data
**Solution:** Transfer Learning with **EfficientNet-B1** pretrained on ImageNet. Heavy data augmentation (random crop, flips, rotation, color jitter) was applied to prevent overfitting.

### Problem 3: Model Generalization
**Solution:** 5-Fold Cross-Validation ensures the model isn't biased towards a specific subset. Model Ensemble combines predictions from all 5 folds via softmax averaging.

---

## Methodology

### 1. Data Preprocessing

**Train Transformations:**
```python
transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

**Validation/Test Transformations:**
```python
transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

### 2. Training Strategy
| Phase | Epochs | Learning Rate | Strategy |
|:--- | :--- | :--- | :--- |
| Feature Extraction | 10 | 1e-4 | Train only classifier (frozen backbone) |
| Fine-tuning | 10 | 1e-5 | Unfreeze all layers, smaller LR |

---

## Results

### Kaggle Scores
| Metric | Score |
|:--- | :--- |
| **Private Score** | **0.8741 (87.41%)** |
| **Public Score** | **0.8733 (87.33%)** |

---

## Project Structure
```text
cassava-leaf-disease/
├── README.md                          # Documentation
├── requirements.txt                   # Dependencies
├── training/
│   └── cassava-leaf_training.ipynb    # 5-fold CV training
├── inference/
    └── inference-cassava.ipynb        # Ensemble inference


```

---

## Installation & Usage

1. **Clone Repository:**
   ```bash
   git clone [https://github.com/VadimTTT/cassava-leaf-disease](https://github.com/VadimTTT/cassava-leaf-disease)
   cd cassava-leaf-disease
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Setup:**
   Place Kaggle data in `kaggle/input/cassava-leaf-disease-classification/`.

4. **Run Inference:**
   Open `inference/inference-cassava.ipynb` and execute to generate `submission.csv`.

---

## Key Learnings
*   **Transfer Learning works:** Pretrained models outperform custom CNNs by over 10%.
*   **CV is essential:** Single splits are unreliable; 5-fold CV provided stable 87%+ accuracy.
*   **Augmentation matters:** It prevents overfitting and adds ~4% to the final score.
*   **Ensemble > Single:** Averaging probabilities smoothed out errors from individual models.

---

## Future Improvements
*   **TTA (Test Time Augmentation):** Average predictions across multiple orientations of test images.
*   **Larger Backbones:** Experimenting with EfficientNet-B4 or Vision Transformers (ViT).
*   **Stacking:** Using a meta-model to weight the importance of each fold's prediction.

---

## Acknowledgments
*   Kaggle for the dataset and competition platform.
*   PyTorch team for the deep learning framework.

---

**Author:** Vadim 
**License:** MIT License  

⭐ *If you found this project useful, consider giving it a star!*