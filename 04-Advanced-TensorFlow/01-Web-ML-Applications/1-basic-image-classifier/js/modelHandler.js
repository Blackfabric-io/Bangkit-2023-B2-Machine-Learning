class ModelHandler {
    constructor() {
        this.model = null;
        this.modelPath = {
            model: 'basic-classifier-model.json',
            weights: 'basic-classifier-weights.bin'
        };
    }

    async loadModel() {
        try {
            this.model = await tf.loadLayersModel(this.modelPath.model);
            console.log('Model loaded successfully');
            return true;
        } catch (error) {
            console.error('Error loading model:', error);
            return false;
        }
    }

    async predict(imageData) {
        if (!this.model) {
            throw new Error('Model not loaded');
        }

        // Convert image data to tensor
        const tensor = tf.browser.fromPixels(imageData)
            .resizeNearestNeighbor([224, 224]) // Resize to model input size
            .toFloat()
            .expandDims();

        // Normalize the image
        const normalized = tensor.div(255.0);

        // Get prediction
        const prediction = await this.model.predict(normalized).data();
        
        // Cleanup tensors
        tensor.dispose();
        normalized.dispose();

        return this.formatPredictions(prediction);
    }

    formatPredictions(predictions) {
        // Format predictions based on your model's output
        // This example assumes binary classification
        const confidence = predictions[0];
        return [{
            class: confidence > 0.5 ? 'Positive' : 'Negative',
            confidence: (confidence * 100).toFixed(2)
        }];
    }
} 