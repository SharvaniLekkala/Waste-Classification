# Waste Classification Using CNN

## Overview
This project focuses on classifying waste into two categories: **Organic** and **Recyclable** using a **Convolutional Neural Network (CNN)**. The dataset used is from Kaggle: `techsash/waste-classification-data`. The implementation includes **data preprocessing, model training, evaluation, and prediction** using TensorFlow and OpenCV.

## Dataset
The dataset consists of:
- **Train Set**: 22,564 images
- **Test Set**: 2,513 images
- **Categories**: Organic Waste, Recyclable Waste

The dataset is loaded using KaggleHub and preprocessed for training.

## Model Architecture
The CNN model consists of:
1. **Three Convolutional Layers** with ReLU activation and MaxPooling.
2. **Flatten Layer** to convert feature maps into a vector.
3. **Fully Connected Layers** with Dropout for regularization.
4. **Sigmoid Activation** for binary classification.
5. **Binary Cross-Entropy Loss** and **Adam Optimizer**.

## Training
- **Batch Size**: 64
- **Epochs**: 15
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy
- **Augmentation**: Image rescaling

## Results
- **Training Accuracy**: 98.61%
- **Validation Accuracy**: 89.65%
- **Final Loss**: 0.6599

Accuracy and loss trends are visualized using Matplotlib.

## Issues
Although the model performs well on the validation set, there are **inconsistencies in predictions**. When a **non-organic image such as a curtain is placed, the model still classifies it as Organic Waste**, indicating possible overfitting or dataset bias.


## Dependencies
- TensorFlow
- OpenCV
- Matplotlib
- Pandas
- NumPy
- KaggleHub

## Future Improvements
- Implementing **data augmentation** to reduce overfitting.



