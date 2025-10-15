# CIFAR10-CNN-Image-Classifier
A Convolutional Neural Network (CNN) built using TensorFlow and Keras for classifying CIFAR-10 images, deployed using Gradio for interactive predictions.

# 🧠 CIFAR-10 Image Classifier using CNN & Gradio
## 🎯 Overview

This project implements a Convolutional Neural Network (CNN) using TensorFlow & Keras to classify images from the CIFAR-10 dataset into 10 categories such as airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

An interactive Gradio interface allows users to upload any image and get real-time classification results.

----

## 🧩 Key Features

- 📦 Trained on the CIFAR-10 dataset (60,000 color images of size 32×32)
- ⚙️ Custom CNN model built from scratch
- 🧠 Achieved ~67% test accuracy after 30 epochs
- 🌐 Gradio web app for image upload and instant predictions
- 🔍 Real-time confidence scores for top-3 predictions

----

## ⚙️ Technologies Used

- TensorFlow & Keras
- NumPy
- Matplotlib
- Gradio
- Python 3.10+

---

## 🧱 Model Architecture
```python
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])
```
- Loss: Sparse Categorical Crossentropy
- Optimizer: Adam
- Metric: Accuracy

---

## 📊 Training Summary

| Metric              | Value |
| ------------------- | ----- |
| Epochs              | 30    |
| Train Accuracy      | 62.7% |
| Validation Accuracy | 69.8% |
| Test Accuracy       | 67.0% |
| Loss                | 0.96  |

---

## 🖼️ Gradio Demo

Once trained, the model is deployed using Gradio:
```python
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(label="Upload an image to classify"),
    outputs=gr.Label(num_top_classes=3, label="Top Predictions"),
    title="CIFAR-10 Image Classifier",
    description="Upload any image, and this CNN model will classify it into one of 10 categories."
)
iface.launch()
```

---

## 🧠 Concepts Covered

- Convolutional Neural Networks (CNNs)
- Convolution, Padding, Strides, Pooling
- LeNet-5 architecture
- CNN vs ANN
- Backpropagation in CNNs
- Data Augmentation
- Pretrained Models & Transfer Learning


