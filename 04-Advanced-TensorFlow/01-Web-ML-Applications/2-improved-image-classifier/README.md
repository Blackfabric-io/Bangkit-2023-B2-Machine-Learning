# Improved Image Classifier

An enhanced web-based image classification application that leverages TensorFlow.js and Convolutional Neural Networks (CNN) to classify fashion items in real-time. This improved version features enhanced model architecture, better preprocessing, and a more robust user interface.

## Overview

This application implements an advanced image classifier that runs entirely in the browser using TensorFlow.js. It uses a CNN architecture trained on the Fashion MNIST dataset to classify clothing items into 10 different categories.

### Key Features

- Multi-class classification (10 categories)
- Real-time inference with WebGL acceleration
- Advanced CNN architecture with 3 convolutional layers
- Grayscale image preprocessing
- Responsive and accessible UI
- Support for both file upload and webcam input
- Confidence score visualization
- Error handling and user feedback

## Technical Implementation

### Model Architecture
- Input: 28x28 grayscale images
- Architecture:
  - Conv2D (16 filters, 3x3 kernel) + ReLU
  - MaxPooling2D (2x2)
  - Conv2D (64 filters, 3x3 kernel) + ReLU
  - MaxPooling2D (2x2)
  - Conv2D (128 filters, 3x3 kernel) + ReLU
  - MaxPooling2D (2x2)
  - Dense (128 units) + ReLU
  - Dense (10 units) + Softmax
- Output: 10 class probabilities

### Classes
1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Project Structure

```
improved-image-classifier/
├── index.html                      # Main HTML entry point
├── styles.css                      # Enhanced styling
├── js/
│   ├── app.js                     # Main application logic
│   ├── config.js                  # Configuration and constants
│   ├── imageProcessor.js          # Image preprocessing
│   ├── modelHandler.js            # TensorFlow.js operations
├── improved-classifier-model.json  # Model architecture
└── improved-classifier-weights.bin # Model weights
```

## Usage

1. **Image Upload**
   ```javascript
   // Example: Process uploaded image
   const imageProcessor = new ImageProcessor(canvas);
   await imageProcessor.processUploadedImage(file);
   ```

2. **Webcam Classification**
   ```javascript
   // Example: Setup webcam
   const success = await imageProcessor.setupWebcam(videoElement);
   if (success) {
     const imageData = imageProcessor.captureWebcamImage();
     const predictions = await modelHandler.predict(imageData);
   }
   ```

3. **Model Prediction**
   ```javascript
   // Example: Get predictions
   const modelHandler = new ModelHandler();
   await modelHandler.loadModel();
   const predictions = await modelHandler.predict(imageData);
   ```

## Performance

- Model Size: ~395KB
- Average Inference Time: ~50ms
- Memory Usage: ~30MB
- Browser Support: Chrome 80+, Firefox 75+, Safari 13+, Edge 80+

## Development

### Prerequisites
- Modern web browser with WebGL support
- Basic understanding of:
  - HTML5/CSS3/ES6+
  - TensorFlow.js
  - Computer Vision basics
  - CNN architectures

### Setup
1. Clone the repository
2. Serve the directory using a local web server
3. Open `index.html` in a supported browser

### Testing
The application includes error handling for:
- Model loading failures
- Image processing errors
- Webcam access issues
- Invalid input formats
- Memory management

## Best Practices Implemented

1. **Performance Optimization**
   - WebGL acceleration
   - Tensor memory management
   - Image preprocessing optimization
   - Efficient DOM updates

2. **User Experience**
   - Real-time feedback
   - Loading states
   - Error messages
   - Responsive design
   - Accessibility features

3. **Code Quality**
   - Modular architecture
   - Clean code principles
   - Error handling
   - Performance considerations
   - Documentation

## References
- [TensorFlow.js Performance Guide](https://www.tensorflow.org/js/guide/platform_environment)
- [Web Performance Optimization](https://developers.google.com/web/fundamentals/performance)
- [Advanced CNN Architectures](https://www.tensorflow.org/js/tutorials/transfer/image_classification)
- [Browser ML Optimization](https://www.tensorflow.org/js/guide/platform_environment)
- [TensorFlow.js Documentation](https://www.tensorflow.org/js)
- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [CNN Architecture Guide](https://cs231n.github.io/convolutional-networks/)
- [Web Neural Networks Guide](https://developers.google.com/machine-learning/guides) 