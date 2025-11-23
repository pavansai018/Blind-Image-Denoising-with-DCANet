# DCANet-TF: Dual CNN with Attention for Blind Image Denoising  
### TensorFlow • COCO 2017 • ShabbyPages Noise Pipeline

> ⚠️ **Notice**  
> This is an **unofficial TensorFlow/Keras re-implementation** of **DCANet** using  
> **COCO 2017** as clean data and **ShabbyPages (Augraphy)** for synthetic noise.  
>  
> • Original DCANet (PyTorch): https://github.com/WenCongWu/DCANet  
> • Original ShabbyPages dataset & pipeline: https://github.com/sparkfish/shabby-pages  
>  
> Please **cite the original authors** if you use this work.

---

## 1. Overview

This repository contains a **TensorFlow implementation of DCANet**, adapted to train on:

- **COCO 2017** as the source of clean natural images  
- **ShabbyPages / Augraphy** as the noise generator to create degraded/noisy versions

The goal: create a **blind image denoising model** capable of removing complex noise patterns beyond simple Gaussian noise.

---

## 2. References & Credits

### DCANet
- Official GitHub: https://github.com/WenCongWu/DCANet  
- Paper: “Dual Convolutional Neural Network with Attention for Image Blind Denoising”, Multimedia Systems, 2024  

**Citation:**
```bibtex
@article{WuL2024,
  title   = {Dual convolutional neural network with attention for image blind denoising},
  author  = {Wencong Wu and others},
  year    = {2024},
  journal = {Multimedia Systems}
}
```

### ShabbyPages / Augraphy
- GitHub: https://github.com/sparkfish/shabby-pages  
- Uses Augraphy to create realistic print/scan/fax distortions.

**Citation:**
```bibtex
@data{ShabbyPages2023,
  title  = {ShabbyPages: A Reproducible Document Denoising and Binarization Dataset},
  year   = {2023},
  author = {The Augraphy Project},
  url    = {https://github.com/sparkfish/shabby-pages}
}
```

---

## 3. Architecture Summary

This TensorFlow re-implementation preserves the key components of DCANet:

### ✔ Noise Estimation Network
Predicts a per-pixel noise level map using a small CNN with tanh output.

### ✔ SCAM (Spatial & Channel Attention Module)
Enhances features using:
- Spatial attention (GAP + GMP + conv)
- Channel attention (GAP + 1×1 bottleneck conv + sigmoid)

### ✔ Dual-Branch CNN

#### 3.1 Upper Sub-Network (U-Net Style)
- Conv → BN → ReLU blocks  
- MaxPool downsampling  
- Bilinear upsampling  
- Skip connections  
- Focuses on **fine details & textures**

#### 3.2 Lower Sub-Network (Dilated Residual Pyramid)
- Dilation rates: 1 → 2 → … → 8  
- Then reverse with skip connections  
- Captures **global context & large receptive fields**

### ✔ Output Fusion
- Residual combination of both branches  
- Final Conv layer → denoised output  
- Outputs:
  - final image
  - conv_tail (intermediate representation)
  - noise_map

---

## 4. Dataset

### 4.1 COCO 2017 (Clean Images)
Download from: https://cocodataset.org

Directory layout:
```
data/
└── coco2017/
    ├── train2017/
    └── val2017/
```

### 4.2 ShabbyPages Noise Generation
ShabbyPages includes:
- `shabbypipeline.py` → defines Augraphy noise/degradation pipeline  
- `get_pipeline()` → loads default pipeline  
- `pipeline.augment(image)` → produces noisy “shabby” version  

We apply ShabbyPages to **COCO images** to simulate:
- print artifacts  
- blur  
- stains  
- folding  
- ink bleeding  
- wrinkles  
- scan/fax noise  
- and more  

This produces **complex realistic noise**, not just simple Gaussian.

Noise can be:
- pre-generated (offline), OR  
- applied inside the TensorFlow data pipeline (recommended)


<img width="832" height="414" alt="image" src="https://github.com/user-attachments/assets/7c062f24-787b-4d66-95c1-49363e47e749" />
<img width="855" height="414" alt="image" src="https://github.com/user-attachments/assets/079f1ba4-d749-4bb7-af7b-c942ae0faa75" />
<img width="849" height="424" alt="image" src="https://github.com/user-attachments/assets/8f44fe1a-b65e-44b7-971c-b0d0f16c4aef" />

---


## 5. Loss Functions

### ✔ Charbonnier Loss
Smooth L1-like, robust to outliers.

### ✔ Edge Loss (Laplacian Pyramid Approximation)
- Gaussian blur  
- Downsample → Upsample  
- Compute band-pass residual  
- Apply Charbonnier on predicted vs true edges

### Combined Loss:
```
Loss = w1 * Charbonnier + w2 * EdgeLoss
```

---

## 6. Learning Rate Schedule

Replicates PyTorch’s:
- GradualWarmupScheduler  
- ReduceLROnPlateau  

TensorFlow version uses:
- Custom warmup callback  
- Keras ReduceLROnPlateau after warmup

---


## 7. Differences from Official DCANet

- Framework: **TensorFlow/Keras** instead of PyTorch  
- Dataset: **COCO 2017** instead of standard denoising datasets  
- Noise: **ShabbyPages Augraphy pipeline** instead of only Gaussian or real-noise benchmarks  
- Training schedule and hyperparameters adapted for this setup  

This is **not** a numerical replication of the paper — it is an engineering adaptation.

---

## 8. License

- ShabbyPages → MIT License  
- DCANet original repo → see DCANet license  
- This TensorFlow re-implementation → MIT License

---

## 9. Acknowledgements

Thanks to:
- **WenCong Wu et al.** for DCANet  
- **Sparkfish & Augraphy** for ShabbyPages noise dataset  
- **COCO dataset** creators  
- TensorFlow/Keras community  

If you use this repo, please **cite the original authors**.

---
