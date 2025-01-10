# MobileNet Transfer Learning Model

A web-based image classification model using transfer learning from MobileNet, optimized for efficient browser-based inference.

## Project Overview
This project demonstrates transfer learning using the pre-trained MobileNet model for efficient image classification in web browsers. It leverages MobileNet's lightweight architecture while adapting it to custom classification tasks.

## Learning Objectives
- Implement transfer learning with pre-trained models
- Fine-tune model layers for custom tasks
- Optimize model deployment for web browsers
- Handle efficient model loading and inference

## Implementation Details
- **Base Model**: MobileNetV2
- **Transfer Learning Strategy**:
  - Freeze base model layers
  - Add custom classification head
  - Fine-tune top layers
- **Training Configuration**:
  - Learning Rate: 0.0001
  - Optimizer: Adam
  - Batch Size: 32
  - Fine-tuning Epochs: 20

## Model Architecture
```
MobileNetV2 Base (frozen)
│
├─ Global Average Pooling
├─ Dense (512, ReLU)
├─ Dropout (0.5)
├─ Dense (num_classes, Softmax)
```

## Results
- Training Accuracy: 96%
- Validation Accuracy: 94%
- Test Accuracy: 93%
- Model Size: 
  - Base Model: ~14MB
  - Custom Layers: ~2MB
  - Total (Compressed): ~6MB
- Inference Time: ~50ms per image

## Features
- Progressive model loading
- Real-time classification
- Webcam support
- Confidence score visualization
- Mobile device optimization

## Usage
```javascript
// Load the model
const model = await tf.loadLayersModel('model/mobilenet_transfer.json');

// Preprocess image
const processedImg = await preprocessImage(imageInput);

// Make prediction
const prediction = await model.predict(processedImg);
```

## Dependencies
- TensorFlow.js v3.x
- MobileNetV2 pre-trained weights
- Modern web browser with WebGL support

## References
- [MobileNet Paper](https://arxiv.org/abs/1704.04861)
- [Transfer Learning Guide](https://www.tensorflow.org/js/tutorials/transfer/image_classification)
- [TensorFlow.js Models](https://github.com/tensorflow/tfjs-models)
- [MobileNet Implementation](https://github.com/tensorflow/tfjs-models/tree/master/mobilenet) 