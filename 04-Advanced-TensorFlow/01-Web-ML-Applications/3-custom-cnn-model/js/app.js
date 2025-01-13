document.addEventListener('DOMContentLoaded', async () => {
    // Initialize components
    const imagePreview = document.getElementById('imagePreview');
    const featureMapCanvas = document.getElementById('featureMapCanvas');
    
    const imageProcessor = new ImageProcessor(imagePreview);
    const modelHandler = new ModelHandler();
    const featureVisualizer = new FeatureVisualizer(featureMapCanvas);
    
    const processingOverlay = document.getElementById('processingOverlay');
    const statusText = document.getElementById('statusText');

    // UI Elements
    const uploadBtn = document.getElementById('uploadBtn');
    const imageUpload = document.getElementById('imageUpload');
    const webcamBtn = document.getElementById('webcamBtn');
    const webcamElement = document.getElementById('webcam');
    const classifyBtn = document.getElementById('classifyBtn');
    const predictionsDiv = document.getElementById('predictions');

    // State
    let isWebcamActive = false;
    let isProcessing = false;

    // Load models
    try {
        setProcessingState(true, 'Loading models...');
        await modelHandler.loadModel();
        setProcessingState(false);
    } catch (error) {
        showError('Failed to load models. Please refresh the page.');
        return;
    }

    // Event Listeners
    uploadBtn.addEventListener('click', () => {
        if (!isProcessing) {
            imageUpload.click();
        }
    });

    imageUpload.addEventListener('change', async (e) => {
        if (e.target.files.length > 0) {
            try {
                await processImage(e.target.files[0]);
                classifyBtn.disabled = false;
                if (isWebcamActive) {
                    stopWebcam();
                }
            } catch (error) {
                showError(error.message);
            }
        }
    });

    webcamBtn.addEventListener('click', async () => {
        if (isProcessing) return;

        try {
            if (!isWebcamActive) {
                const success = await startWebcam();
                if (success) {
                    isWebcamActive = true;
                    webcamElement.style.display = 'block';
                    webcamBtn.textContent = 'Stop Webcam';
                    classifyBtn.disabled = false;
                }
            } else {
                stopWebcam();
            }
        } catch (error) {
            showError('Failed to access webcam');
        }
    });

    classifyBtn.addEventListener('click', async () => {
        if (isProcessing) return;
        await classify();
    });

    // Helper functions
    async function processImage(file) {
        setProcessingState(true, 'Processing image...');
        try {
            await imageProcessor.processUploadedImage(file);
            featureVisualizer.clear();
        } finally {
            setProcessingState(false);
        }
    }

    async function startWebcam() {
        setProcessingState(true, 'Starting webcam...');
        try {
            return await imageProcessor.setupWebcam(webcamElement);
        } finally {
            setProcessingState(false);
        }
    }

    function stopWebcam() {
        imageProcessor.stopWebcam();
        isWebcamActive = false;
        webcamElement.style.display = 'none';
        webcamBtn.textContent = 'Use Webcam';
        classifyBtn.disabled = true;
        featureVisualizer.clear();
    }

    async function classify() {
        setProcessingState(true, 'Running inference...');

        try {
            let imageData;
            if (isWebcamActive) {
                imageData = imageProcessor.captureWebcamImage();
            } else {
                imageData = imageProcessor.getCanvas();
            }

            // Get predictions and feature map
            const [predictions, featureMap] = await Promise.all([
                modelHandler.predict(imageData),
                modelHandler.getFeatureMap(imageData)
            ]);

            // Display results
            displayPredictions(predictions);
            featureVisualizer.visualizeFeatureMap(featureMap);
        } catch (error) {
            console.error('Classification error:', error);
            showError('Classification failed. Please try again.');
        } finally {
            setProcessingState(false);
        }
    }

    function displayPredictions(predictions) {
        predictionsDiv.innerHTML = '';
        
        predictions.forEach(pred => {
            const predictionItem = document.createElement('div');
            predictionItem.className = 'prediction-item';
            
            const label = document.createElement('span');
            label.textContent = pred.class;
            
            const confidence = document.createElement('span');
            confidence.textContent = `${pred.confidence}%`;
            
            const bar = document.createElement('div');
            bar.className = 'confidence-bar';
            
            // Animate the confidence bar
            setTimeout(() => {
                bar.style.width = `${pred.confidence}%`;
            }, 50);

            predictionItem.appendChild(label);
            predictionItem.appendChild(confidence);
            predictionItem.appendChild(bar);
            predictionsDiv.appendChild(predictionItem);
        });
    }

    function setProcessingState(processing, message = 'Processing...') {
        isProcessing = processing;
        processingOverlay.style.display = processing ? 'flex' : 'none';
        statusText.textContent = message;
        classifyBtn.disabled = processing;
    }

    function showError(message) {
        console.error(message);
        alert(message);
        setProcessingState(false);
    }
}); 