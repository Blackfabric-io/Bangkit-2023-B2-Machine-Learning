class ModelHandler {
    constructor() {
        this.model = null;
        this.isLoading = false;
    }

    async loadModel() {
        if (this.isLoading) return false;
        
        try {
            this.isLoading = true;
            this.model = await tf.loadLayersModel(CONFIG.model.path.json);
            console.log('Model loaded successfully');
            return true;
        } catch (error) {
            console.error('Error loading model:', error);
            return false;
        } finally {
            this.isLoading = false;
        }
    }

    async predict(imageData) {
        if (!this.model) {
            throw new Error('Model not loaded');
        }

        return tf.tidy(() => {
            // Convert image data to tensor
            let tensor = tf.browser.fromPixels(imageData);
            
            // Convert to grayscale if needed
            if (CONFIG.image.grayscale) {
                tensor = tf.mean(tensor, 2).expandDims(2);
            }
            
            // Resize to model input size
            tensor = tf.image.resizeBilinear(tensor, CONFIG.model.inputShape);
            
            // Normalize and expand dimensions
            tensor = tensor.toFloat().div(255.0).expandDims(0);

            // Get prediction
            const predictions = this.model.predict(tensor);
            const probabilities = predictions.dataSync();
            
            return this.formatPredictions(probabilities);
        });
    }

    formatPredictions(probabilities) {
        // Convert to array of {class, confidence} objects
        const predictions = Array.from(probabilities).map((confidence, index) => ({
            class: CONFIG.model.classes[index],
            confidence: (confidence * 100).toFixed(2)
        }));

        // Sort by confidence and get top K predictions
        return predictions
            .sort((a, b) => b.confidence - a.confidence)
            .filter(pred => pred.confidence > CONFIG.prediction.confidenceThreshold)
            .slice(0, CONFIG.prediction.topK);
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
    }
} 