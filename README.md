# medical-image-segmentation-uncertainty
Comparative study of loss functions for uncertainty-aware medical image segmentation using U-Net

# Uncertainty Quantification in Medical Image Segmentation

A comparative study of loss functions for pancreas and spleen segmentation using a U-Net architecture, with uncertainty estimation via Monte Carlo Dropout.

## 🩺 Problem Statement

Accurate segmentation of medical images (like CT scans) is critical for diagnosis and treatment planning. But accuracy alone isn't enough for clinical trust — models also need to communicate **how confident** they are in each prediction. This project investigates how different loss functions affect both segmentation accuracy and the reliability of uncertainty estimates, using the pancreas and spleen datasets from the Medical Segmentation Decathlon (MSD).

## 🎯 Objectives

- Train a U-Net segmentation model on pancreas and spleen CT scans
- Compare 7 loss functions: **Dice, BCE, Log-Cosh, Tversky, Focal Tversky, Log Dice, and a Hybrid loss (Jaccard + BCE + SSIM)**
- Quantify prediction uncertainty using **Monte Carlo Dropout**
- Evaluate robustness across different learning rates and training epochs

## 🧠 Methodology

1. **Data**: 3D CT volumes (NIfTI format) from MSD, sliced into 2D axial images
2. **Preprocessing**: Hounsfield Unit (HU) intensity windowing + median filtering for noise reduction
3. **Model**: U-Net (encoder-decoder with skip connections)
4. **Training**: Adam optimizer, learning rate scheduling, early stopping on validation Dice
5. **Uncertainty Estimation**: Monte Carlo Dropout (T=10 stochastic forward passes) with entropy-based pixel-wise uncertainty
6. **Evaluation**: Dice Coefficient, Uncertainty Mean/Std, Mean Prediction Mean/Std

## 📊 Key Results

| Loss Function | Pancreas Dice | Spleen Dice |
|---|---|---|
| **Dice Loss** | **0.684** | **0.847** |
| Focal Tversky | 0.670 | 0.895 |
| Tversky | 0.678 | 0.838 |
| Log Dice | 0.668 | 0.833 |
| BCE | 0.627 | 0.793 |
| Hybrid | 0.568 | 0.845 |
| Log-Cosh | 0.475 | 0.464 |

**Finding**: Dice Loss consistently delivered the best balance of high segmentation accuracy *and* low, well-calibrated uncertainty across both datasets and multiple training regimes (varying learning rates and epochs). This makes it the most robust choice for clinical-grade segmentation models.

See `/notebooks` for full training curves, violin plots of uncertainty distributions, and qualitative segmentation visualizations.

## 🛠️ Tech Stack

- Python, TensorFlow/Keras
- NumPy, Matplotlib, Seaborn (visualization)
- NiBabel (NIfTI file handling)
- Monte Carlo Dropout for uncertainty estimation

## 📁 Repository Structure

```
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_unet_training.ipynb
│   ├── 03_loss_function_comparison.ipynb
│   ├── 04_uncertainty_quantification.ipynb
│   └── 05_results_visualization.ipynb
├── results/
│   ├── dice_coefficient_plots/
│   ├── uncertainty_violin_plots/
│   └── segmentation_samples/
├── paper/
│   └── Uncertainty_Quantification_in_Medical_Image_Segmentation.pdf
├── requirements.txt
└── README.md
```

## 🚀 How to Run

```bash
git clone https://github.com/<your-username>/medical-image-segmentation-uncertainty.git
cd medical-image-segmentation-uncertainty
pip install -r requirements.txt
```

Open any notebook in `/notebooks` and run cells sequentially. Dataset can be downloaded from the [Medical Segmentation Decathlon](http://medicaldecathlon.com/).

## 📄 Reference

This project reproduces and extends the methodology described in the accompanying paper (see `/paper`), which benchmarks 7 loss functions for uncertainty-aware medical image segmentation.

## 📌 Future Work

- Extend to 3D volumetric segmentation
- Integrate uncertainty maps into a clinician-facing dashboard
- Test on additional MSD organ datasets

---
*This project was built to explore practical applications of deep learning in medical imaging, with a focus on model trustworthiness — a critical requirement for real-world clinical deployment.*
