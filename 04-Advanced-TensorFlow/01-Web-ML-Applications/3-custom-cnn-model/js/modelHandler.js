class ModelHandler {
    constructor() {
        this.customModel = null;
        this.baseModel = null;
        this.isLoading = false;
    }

    async loadModel() {
        if (this.isLoading) return false;
        
        try {
            this.isLoading = true;
            
            // Load MobileNet base model
            this.baseModel = await mobilenet.load({
                version: 1,
                alpha: 1.0
            });
            
            // Load custom top layers
            this.customModel = await tf.loadLayersModel(CONFIG.model.path.json);
            
            console.log('Models loaded successfully');
            return true;
        } catch (error) {
            console.error('Error loading models:', error);
            return false;
        } finally {
            this.isLoading = false;
        }
    }

    async predict(imageData) {
        if (!this.baseModel || !this.customModel) {
            throw new Error('Models not loaded');
        }

        return tf.tidy(() => {
            // Preprocess image and get base features
            const processedTensor = this.preprocessImage(imageData);
            const features = this.extractFeatures(processedTensor);
            
            // Get predictions from custom model
            const predictions = this.customModel.predict(features);
            const probabilities = predictions.dataSync();
            
            return this.formatPredictions(probabilities);
        });
    }

    preprocessImage(imageData) {
        // Convert to tensor and resize
        let tensor = tf.browser.fromPixels(imageData)
            .resizeBilinear(CONFIG.model.inputShape);

        // Normalize if needed
        if (CONFIG.image.normalize) {
            tensor = tf.sub(tensor, CONFIG.image.meanRGB);
        }

        // Expand dimensions for batch
        return tensor.expandDims(0);
    }

    extractFeatures(inputTensor) {
        // Get intermediate layer activation
        const activation = this.baseModel.model.execute(
            inputTensor,
            CONFIG.model.featureLayer
        );

        return activation;
    }

    async getFeatureMap(imageData) {
        return tf.tidy(() => {
            const processedTensor = this.preprocessImage(imageData);
            const features = this.extractFeatures(processedTensor);
            
            // Get feature map data
            const featureData = features.dataSync();
            const [_, height, width, channels] = features.shape;
            
            // Average across channels for visualization
            const avgFeatures = new Float32Array(height * width);
            for (let i = 0; i < height * width; i++) {
                let sum = 0;
                for (let c = 0; c < channels; c++) {
                    sum += featureData[i * channels + c];
                }
                avgFeatures[i] = sum / channels;
            }
            
            return {
                data: avgFeatures,
                height,
                width
            };
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
            .filter(pred => pred.confidence > CONFIG.prediction.confidenceThreshold * 100)
            .slice(0, CONFIG.prediction.topK);
    }

    dispose() {
        if (this.customModel) {
            this.customModel.dispose();
            this.customModel = null;
        }
        if (this.baseModel) {
            this.baseModel.dispose();
            this.baseModel = null;
        }
    }
} 