# 🐶 **Dog Breed Classification using Transfer Learning – Deep Learning Project**

This project classifies dog breeds using deep learning models based on transfer learning. It aims to automate pet identification, support veterinary diagnostics, and enhance dog breed recognition in real-world applications with high accuracy.

🔬 Built as part of an academic project by Mopada Surya Prakash, UG Scholar at Mohan Babu University, Tirupati, India.

---
![image](https://github.com/user-attachments/assets/9999c82a-69bb-4298-a436-18dbc37c9bb1)
---

## 🧭 Project Overview

**Problem Statement:** Identifying dog breeds manually is subjective, error-prone, and time-consuming. With over 120 recognized breeds, accurate identification requires expert knowledge.  

**Solution:** This project uses pre-trained convolutional neural networks (CNNs) with transfer learning to classify multiple dog breeds with minimal training data.  

**Use Case:** Useful in pet adoption agencies, veterinary clinics, animal shelters, and pet identification apps.

---

## 📂 Files Included

- `dog_breed_classification.ipynb`: Main notebook for training and evaluating the model  
- `dataset/`: Folder containing images of labeled dog breeds  
- `outputs/`: Contains results, visualizations, and performance metrics  
- `README.md`: This documentation  

---

## ✨ Model Highlights

- **Base Model:** InceptionResNetV2 (pretrained on ImageNet)  
- **Classification Head:** Dense → Dropout → Dense → Softmax  
- **Data Augmentation:** Real-time transformations  
- **Optimizer:** Adam  
- **Loss Function:** Sparse Categorical Crossentropy  
- **Evaluation:** Accuracy and loss on training and validation sets  

---

## 🧪 Dataset

- **Source:** Stanford Dogs Dataset  
- **Classes:** Multiple dog breeds (e.g., Afghan Hound, Beagle, Spaniel)  
- **Format:** Images categorized into subfolders by breed  
- **Split:** Train, Validation, Test folders  

---

## 🧹 Preprocessing Pipeline

- Image resizing to (224, 224)  
- Normalization and augmentation  
- Train-Validation-Test splitting  
- Batch processing using `ImageDataGenerator`  
- Categorical label encoding  

---

## 📊 Training Performance

| Metric               | Value     |
|----------------------|-----------|
| **Train Accuracy**   | ~95.38%   |
| **Validation Accuracy** | ~93.42%   |
| **Train Loss**       | ~0.18     |
| **Validation Loss**  | ~0.23     |

> The model shows high training and validation accuracy with minimal loss, proving its generalization capability.

---

## 🧱 Model Architecture

- InceptionResNetV2 Backbone  
- Global Average Pooling  
- Dense(512) → Dropout(0.3) → Dense(256) → Dropout(0.2)  
- Final Dense layer with softmax activation for classification  

---

## 🛠️ Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy, Pandas  
- Matplotlib, Seaborn  
- OpenCV, PIL  
- Jupyter Notebook / Google Colab  

---
