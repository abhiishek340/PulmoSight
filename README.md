# 🫁 PneumoScanAI – Deep Learning for Pneumonia Detection from NIH Chest X-rays

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success)

**PneumoScanAI** is a deep learning-based web application designed to detect **pneumonia from chest X-ray images** using the large-scale **NIH ChestX-ray14** dataset. The solution applies transfer learning on the VGG16 architecture, fine-tuned to distinguish pneumonia cases with high confidence. It includes a real-time prediction API using Flask.

---

## 📚 Dataset: NIH ChestX-ray14

- **Source**: [NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- **Total Images**: 112,120
- **Patients**: Over 30,000 unique individuals
- **Labels**: 14 disease labels including:
  - Pneumonia
  - Infiltration
  - Effusion
  - Mass
  - Atelectasis
- **Format**: JPEG, labeled via a CSV metadata file (`Data_Entry_2017.csv`)
- **Size**: ~40 GB

> **Note**: Preprocessing included label filtering (Pneumonia vs. Normal), class balancing, and resizing to 224x224 pixels.

---

## 🧠 Model Architecture

```text
Input Image (224x224x3)
      ↓
Pretrained VGG16 (imagenet weights, frozen base)
      ↓
GlobalAveragePooling2D
      ↓
Dense(128, ReLU) + Dropout(0.5)
      ↓
Dense(1, Sigmoid)
      ↓
Output → Pneumonia / Normal
