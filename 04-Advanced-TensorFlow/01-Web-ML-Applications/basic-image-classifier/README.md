# Basic Image Classifier

A foundational web-based image classification project using TensorFlow.js that introduces core concepts of in-browser machine learning.

## Project Overview
This project implements a basic image classifier that runs directly in the web browser using TensorFlow.js. It serves as an introduction to browser-based ML inference and model deployment.

## Learning Objectives
- Set up TensorFlow.js in a web application
- Load and preprocess images in the browser
- Implement basic image classification inference
- Handle model loading and prediction efficiently

## Implementation Details
- **Framework**: TensorFlow.js
- **Model Architecture**: Simple CNN with 2-3 convolutional layers
- **Dataset**: MNIST/CIFAR-10 for initial testing
- **Preprocessing**: Basic image normalization and resizing
- **Deployment**: Static web hosting with CDN-served model

## Results
- Training Accuracy: ~85%
- Validation Accuracy: ~82%
- Inference Time: ~150ms per image
- Model Size: ~5MB

## Usage
1. Open `index.html` in a modern web browser
2. Upload an image or use the webcam
3. Click classify to get predictions
4. View confidence scores for each class

## Dependencies
- TensorFlow.js v3.x
- Modern web browser with WebGL support
- Basic HTML/CSS/JavaScript

## References
- [TensorFlow.js Tutorials](https://www.tensorflow.org/js/tutorials)
- [Browser ML Guide](https://developers.google.com/web/updates/capabilities)
- [Image Classification Tutorial](https://www.tensorflow.org/js/tutorials/transfer/image_classification) 