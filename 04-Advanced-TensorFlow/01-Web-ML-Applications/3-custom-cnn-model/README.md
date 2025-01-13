# Custom CNN Image Classifier

A custom-designed web-based image classification application that combines MobileNet feature extraction with a custom CNN architecture for transfer learning. This implementation showcases advanced deep learning concepts with real-time feature visualization.

## Overview

This application demonstrates transfer learning by utilizing MobileNet as a feature extractor and adding custom dense layers for specific image classification tasks. It features:

- Transfer learning with MobileNet
- Custom CNN architecture for classification
- Real-time feature map visualization
- Advanced image preprocessing
- WebGL-accelerated inference
- Responsive and accessible UI

## Technical Implementation

### Model Architecture
1. **Base Model (MobileNet)**
   - Pre-trained on ImageNet
   - Feature extraction from layer 'conv_pw_13_relu'
   - Input: 224x224x3 RGB images

2. **Custom Layers**
   - Flatten layer for 7x7x1024 features
   - Dense layer (100 units) with ReLU
   - Output layer (5 units) with Softmax

### Classes
1. Animal
2. Landscape
3. Person
4. Vehicle
5. Building

## Project Structure

```
custom-cnn-model/
├── index.html                  # Main application entry
├── styles.css                 # Enhanced UI styling
├── js/
│   ├── app.js                # Application logic
│   ├── config.js             # Configuration settings
│   ├── imageProcessor.js     # Image preprocessing
│   ├── modelHandler.js       # Model operations
│   ├── featureVisualizer.js  # Feature map visualization
├── custom-cnn-model.json     # Model architecture
└── custom-cnn-weights.bin    # Model weights
```
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

## Usage Examples

1. **Model Initialization**
```javascript
const modelHandler = new ModelHandler();
await modelHandler.loadModel();
```

2. **Image Processing**
```javascript
const imageProcessor = new ImageProcessor(canvas);
await imageProcessor.processUploadedImage(file);
```

3. **Feature Visualization**
```javascript
const featureMap = await modelHandler.getFeatureMap(imageData);
featureVisualizer.visualizeFeatureMap(featureMap);
```

## Performance Metrics

- Model Size: ~19MB
- Average Inference Time: ~100ms
- Memory Usage: ~50MB
- Feature Extraction Time: ~30ms

## Development Setup

### Prerequisites
- Modern web browser with WebGL 2.0
- Understanding of:
  - TensorFlow.js
  - Transfer Learning
  - CNN Architectures
  - WebGL

### Installation
1. Clone the repository
2. Serve the directory using a web server
3. Open `index.html` in a browser

## Testing Framework

The application includes comprehensive error handling and validation:

1. **Model Loading**
   - Base model verification
   - Custom layer compatibility
   - Weight loading validation

2. **Image Processing**
   - Input validation
   - Size constraints
   - Format checking
   - Memory management

3. **Feature Visualization**
   - WebGL context validation
   - Canvas support checking
   - Memory cleanup

## Best Practices

1. **Performance**
   - WebGL acceleration
   - Batch processing
   - Memory management
   - Tensor disposal
   - Async operations

2. **User Experience**
   - Progressive loading
   - Real-time feedback
   - Error handling
   - Responsive design
   - Accessibility

3. **Code Quality**
   - Modular architecture
   - Clean code principles
   - Documentation
   - Type checking
   - Error boundaries

## References
- [CNN Architecture Guide](https://www.tensorflow.org/js/tutorials/training/handwritten_digit_cnn)
- [TensorFlow.js Model Optimization](https://www.tensorflow.org/js/guide/platform_environment)
- [Custom Layers API](https://www.tensorflow.org/js/guide/layers_for_keras_users)
- [Web ML Performance Best Practices](https://developers.google.com/web/updates/capabilities) 
- [TensorFlow.js Transfer Learning](https://www.tensorflow.org/js/tutorials/transfer/image_classification)
- [MobileNet Architecture](https://arxiv.org/abs/1704.04861)
- [Feature Visualization Techniques](https://distill.pub/2017/feature-visualization/)
- [WebGL Performance Guide](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_best_practices) 