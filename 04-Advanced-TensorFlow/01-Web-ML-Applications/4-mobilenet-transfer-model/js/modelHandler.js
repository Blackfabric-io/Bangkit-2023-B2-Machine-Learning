class ModelHandler {
    constructor() {
        this.mobilenetModel = null;
        this.customModel = null;
        this.activeModel = 'mobilenet';
        this.selectedLayer = 'conv_pw_13_relu';
        this.isLoading = false;
        this.performanceStats = {
            inferenceTime: 0,
            memoryUsage: 0
        };
    }

    async loadModels() {
        if (this.isLoading) return false;
        
        try {
            this.isLoading = true;
            
            // Load MobileNet base model
            this.mobilenetModel = await mobilenet.load({
                version: CONFIG.model.mobilenet.version,
                alpha: CONFIG.model.mobilenet.alpha
            });
            
            // Load custom fine-tuned model
            this.customModel = await tf.loadLayersModel(CONFIG.model.custom.path.json);
            
            console.log('Models loaded successfully');
            return true;
        } catch (error) {
            console.error('Error loading models:', error);
            return false;
        } finally {
            this.isLoading = false;
        }
    }

    setActiveModel(modelType) {
        if (modelType !== 'mobilenet' && modelType !== 'custom') {
            throw new Error('Invalid model type');
        }
        this.activeModel = modelType;
    }

    setFeatureLayer(layerName) {
        if (!CONFIG.model.featureLayers[layerName]) {
            throw new Error('Invalid feature layer');
        }
        this.selectedLayer = layerName;
    }

    async predict(imageData) {
        if (!this.mobilenetModel || !this.customModel) {
            throw new Error('Models not loaded');
        }

        const startTime = performance.now();
        const memoryBefore = tf.memory().numBytes;

        try {
            return await tf.tidy(() => {
                // Preprocess image
                const processedTensor = this.preprocessImage(imageData);
                
                // Get predictions based on active model
                let predictions;
                if (this.activeModel === 'mobilenet') {
                    predictions = this.mobilenetModel.classify(processedTensor);
                } else {
                    const features = this.extractFeatures(processedTensor);
                    predictions = this.customModel.predict(features);
                    predictions = this.formatCustomPredictions(predictions);
                }

                return predictions;
            });
        } finally {
            // Update performance metrics
            this.performanceStats.inferenceTime = performance.now() - startTime;
            this.performanceStats.memoryUsage = (tf.memory().numBytes - memoryBefore) / (1024 * 1024);
        }
    }

    preprocessImage(imageData) {
        // Convert to tensor and resize
        let tensor = tf.browser.fromPixels(imageData)
            .resizeBilinear(CONFIG.model.mobilenet.inputShape);

        // Normalize if needed
        if (CONFIG.image.normalize) {
            tensor = tf.sub(tensor, CONFIG.image.meanRGB);
            tensor = tf.div(tensor, 255);
        }

        return tensor.expandDims(0);
    }

    extractFeatures(inputTensor) {
        // Get intermediate layer activation
        const activation = this.mobilenetModel.model.execute(
            inputTensor,
            this.selectedLayer
        );

        return activation;
    }

    async getFeatureMaps(imageData) {
        return tf.tidy(() => {
            const processedTensor = this.preprocessImage(imageData);
            const features = this.extractFeatures(processedTensor);
            
            // Get feature map data
            const featureData = features.dataSync();
            const [_, height, width, channels] = features.shape;
            
            // Select subset of channels for visualization
            const selectedChannels = this.selectFeatureChannels(channels);
            
            // Extract feature maps for selected channels
            const featureMaps = selectedChannels.map(channel => {
                const map = new Float32Array(height * width);
                for (let i = 0; i < height * width; i++) {
                    map[i] = featureData[i * channels + channel];
                }
                return {
                    data: map,
                    height,
                    width,
                    channel
                };
            });
            
            return featureMaps;
        });
    }

    selectFeatureChannels(totalChannels) {
        const maxMaps = CONFIG.visualization.maxFeatureMaps;
        const step = Math.floor(totalChannels / maxMaps);
        const channels = [];
        
        for (let i = 0; i < maxMaps; i++) {
            channels.push(i * step);
        }
        
        return channels;
    }

    formatCustomPredictions(predictions) {
        const probabilities = predictions.dataSync();
        return Array.from(probabilities).map((confidence, index) => ({
            className: CONFIG.prediction.classNames[index],
            probability: confidence
        }))
        .sort((a, b) => b.probability - a.probability)
        .filter(pred => pred.probability > CONFIG.prediction.confidenceThreshold)
        .slice(0, CONFIG.prediction.topK);
    }

    getPerformanceStats() {
        return this.performanceStats;
    }

    dispose() {
        if (this.mobilenetModel) {
            this.mobilenetModel.dispose();
            this.mobilenetModel = null;
        }
        if (this.customModel) {
            this.customModel.dispose();
            this.customModel = null;
        }
    }
} 