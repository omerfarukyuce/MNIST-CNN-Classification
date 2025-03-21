# 🔢 MNIST Digit Classification with Convolutional Neural Network 🤖

## 📝 Project Overview
This project implements a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset. The model achieves high accuracy in recognizing and classifying digits from 0 to 9.

## ✨ Key Features
- 🧠 Utilizes TensorFlow and Keras for deep learning
- 🏗️ Implements a CNN architecture with multiple convolutional and dense layers
- 🔄 Includes data augmentation using ImageDataGenerator
- 📊 Achieves over 99% accuracy on the MNIST test dataset

## 🏛️ Model Architecture
The neural network consists of:
- 🌐 Two sets of Convolutional and MaxPooling layers
- 📈 Batch Normalization and Dropout for regularization
- 🎯 Dense layers with ReLU and Softmax activations

## 📊 Performance Metrics
- **Test Accuracy**: 99.44% 🏆
- **Test Precision**: 99.48% 🎯
- **Test Recall**: 99.44% 🚀

## 🛠️ Dependencies
- 🐍 Python 3.7+
- 🤖 TensorFlow 2.x
- 📊 NumPy
- 🐼 Pandas
- 📈 Matplotlib
- 🌈 Seaborn
- 🧮 Scikit-learn

## 🚀 Usage
# Load the pre-trained model
from tensorflow.keras.models import load_model
model = load_model('mnist_model.h5')

# Predict on a new image
prediction = model.predict(your_image)

## 📂 Project Structure
mnist_model.h5: 🧠 Trained neural network model
mnist_digit_classification.py: 📝 Main script with model training and evaluation


## 📊 Visualization
The project includes various visualizations:
📈 Training and validation accuracy/loss curves
🕵️ Misclassified samples
🔍 Confusion matrix
📊 Classification accuracy per digit
