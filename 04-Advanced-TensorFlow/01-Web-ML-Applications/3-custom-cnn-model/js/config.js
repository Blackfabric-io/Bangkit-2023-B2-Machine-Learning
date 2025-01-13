const CONFIG = {
    model: {
        path: {
            json: 'custom-cnn-model.json',
            weights: 'custom-cnn-weights.bin'
        },
        inputShape: [224, 224],
        classes: [
            'Animal',
            'Landscape',
            'Person',
            'Vehicle',
            'Building'
        ],
        featureLayer: 'conv_pw_13_relu'  // MobileNet feature extraction layer
    },
    image: {
        maxSize: 224,
        normalize: true,
        meanRGB: [123.68, 116.779, 103.939]
    },
    prediction: {
        topK: 2,
        confidenceThreshold: 0.15,
        batchSize: 1
    },
    visualization: {
        featureMapSize: 200,
        colormap: 'viridis',
        alpha: 0.6
    }
}; 