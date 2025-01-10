# Fashion MNIST TFLite

A mobile-optimized implementation of the Fashion MNIST classifier using TensorFlow Lite, designed for efficient on-device inference.

## Project Overview
This project converts and optimizes a Fashion MNIST classifier for mobile deployment using TensorFlow Lite. It demonstrates mobile-specific optimizations while maintaining high accuracy.

## Learning Objectives
- Convert TensorFlow models to TFLite format
- Implement quantization for model optimization
- Deploy models on mobile devices
- Optimize inference performance

## Implementation Details
- **Original Model**: CNN trained on Fashion MNIST
- **Optimization Techniques**:
  - Post-training quantization
  - Weight pruning
  - Layer optimization
- **Mobile Integration**:
  - Custom Android/iOS delegates
  - Memory optimization
  - Battery-efficient inference

## Model Architecture
```
Input (28x28x1)
│
├─ Conv2D (32 filters, 3x3)
├─ MaxPooling (2x2)
│
├─ Conv2D (64 filters, 3x3)
├─ MaxPooling (2x2)
│
├─ Conv2D (64 filters, 3x3)
│
├─ Flatten
├─ Dense (64)
├─ Dense (10)
```

## Results
- Original Model Accuracy: 91.0%
- TFLite Model Accuracy: 90.98%
- Model Size:
  - Original: 28MB
  - Quantized: 7MB
- Inference Time:
  - CPU: 45ms
  - GPU Delegate: 15ms
  - NNAPI: 12ms

## Features
- Real-time classification
- Camera integration
- Batch processing support
- Performance monitoring
- Cross-platform support

## Integration Guide
1. Add TFLite dependency:
```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.5.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.5.0'
}
```

2. Load the model:
```java
Interpreter tflite = new Interpreter(loadModelFile(context));
```

3. Run inference:
```java
tflite.run(inputArray, outputArray);
```

## Dependencies
- TensorFlow Lite 2.5.0+
- Android API 21+ / iOS 12+
- OpenGL ES 3.1+
- Camera2 API (optional)

## References
- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [TFLite Model Optimization](https://www.tensorflow.org/lite/performance/model_optimization)
- [TFLite Android Guide](https://www.tensorflow.org/lite/guide/android)
- [TFLite iOS Guide](https://www.tensorflow.org/lite/guide/ios) 