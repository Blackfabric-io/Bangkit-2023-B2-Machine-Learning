const CONFIG = {
    model: {
        path: {
            json: 'improved-classifier-model.json',
            weights: 'improved-classifier-weights.bin'
        },
        inputShape: [28, 28],
        classes: [
            'T-shirt/top',
            'Trouser',
            'Pullover',
            'Dress',
            'Coat',
            'Sandal',
            'Shirt',
            'Sneaker',
            'Bag',
            'Ankle boot'
        ]
    },
    image: {
        maxSize: 400,
        grayscale: true
    },
    prediction: {
        topK: 3,
        confidenceThreshold: 0.1
    }
}; 