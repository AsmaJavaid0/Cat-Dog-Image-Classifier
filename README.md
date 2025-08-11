# 🐱🐶 Cat vs. Dog Image Classification using Transfer Learning (VGG16)

## 📌 Overview
This project trains a deep learning model to classify images as **Cat** or **Dog** using **Transfer Learning** with the VGG16 architecture.  
We use a dataset stored in the `PetImages` folder, split it into **training** and **testing** sets, clean corrupted images, and then train the model.

---

## 📂 Project Structure
PetImages/ # Original dataset
│── Cat/ # Cat images
│── Dog/ # Dog images
petimages_split/ # Cleaned & split dataset
│── train/
│ ├── Cat/
│ └── Dog/
│── test/
├── Cat/
└── Dog/

---

## ⚙️ Steps

### 1️⃣ Dataset Preparation
- Load original **PetImages** dataset.
- Remove corrupted images using Pillow.
- Split into **80% training** and **20% testing** sets.

### 2️⃣ Data Loading
We use **Keras `ImageDataGenerator`** for:
- Rescaling pixel values (`rescale=1./255`)
- Loading images from directory
- Batching and shuffling

### 3️⃣ Model Architecture
We use **VGG16** (pre-trained on ImageNet) with:
- **Frozen base layers** (transfer learning)
- Flatten layer
- Dense layer (256 units, ReLU activation)
- Dropout (0.5)
- Final Dense layer with 2 units (softmax)

### 4️⃣ Training
```python
history_feat = model_feature_extract.fit(train_generator, epochs=15)
Optimizer: Adam(learning_rate=0.001)

Loss: categorical_crossentropy

Metric: accuracy
##🛠 Requirements
bash
Copy
Edit
tensorflow
keras
pillow
numpy

