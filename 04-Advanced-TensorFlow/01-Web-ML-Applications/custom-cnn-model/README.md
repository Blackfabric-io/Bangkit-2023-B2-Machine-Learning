# Custom CNN Model

A custom-designed Convolutional Neural Network (CNN) implemented in TensorFlow.js for web-based image classification tasks.

## Project Overview
This project implements a custom CNN architecture optimized for web deployment. It demonstrates how to design, train, and deploy custom neural networks using TensorFlow.js.

## Learning Objectives
- Design custom CNN architectures
- Implement advanced training techniques
- Optimize model parameters for web deployment
- Balance model complexity and performance

## Model Architecture
```
Input Layer (224x224x3)
│
├─ Conv2D (64 filters, 3x3, ReLU)
├─ MaxPooling (2x2)
├─ BatchNormalization
│
├─ Conv2D (128 filters, 3x3, ReLU)
├─ MaxPooling (2x2)
├─ BatchNormalization
│
├─ Conv2D (256 filters, 3x3, ReLU)
├─ MaxPooling (2x2)
├─ BatchNormalization
│
├─ Flatten
├─ Dense (512, ReLU)
├─ Dropout (0.5)
├─ Dense (num_classes, Softmax)
```

## Implementation Details
- **Framework**: TensorFlow.js
- **Training Strategy**:
  - Learning Rate: 0.001 with Adam optimizer
  - Batch Size: 32
  - Epochs: 50
  - Early Stopping: patience=5
- **Data Augmentation**:
  - Random rotation (±15°)
  - Random zoom (±10%)
  - Horizontal flip

## Results
- Training Accuracy: 94%
- Validation Accuracy: 91%
- Test Accuracy: 90%
- Model Size: 12MB (4MB compressed)
- Inference Time: ~100ms per image

## Usage
1. Include the model in your web project:
```html
<script src="model.js"></script>
```
2. Load the model:
```javascript
const model = await tf.loadLayersModel('model/model.json');
```
3. Make predictions:
```javascript
const prediction = await model.predict(preprocessedImage);
```

## Dependencies
- TensorFlow.js v3.x
- Modern web browser with WebGL support
- Node.js v14+ (for training)

## References
- [CNN Architecture Guide](https://www.tensorflow.org/js/tutorials/training/handwritten_digit_cnn)
- [TensorFlow.js Model Optimization](https://www.tensorflow.org/js/guide/platform_environment)
- [Custom Layers API](https://www.tensorflow.org/js/guide/layers_for_keras_users)
- [Web ML Performance Best Practices](https://developers.google.com/web/updates/capabilities) 