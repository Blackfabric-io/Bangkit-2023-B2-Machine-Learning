document.addEventListener('DOMContentLoaded', async () => {
    // Initialize components
    const imagePreview = document.getElementById('imagePreview');
    const imageProcessor = new ImageProcessor(imagePreview);
    const modelHandler = new ModelHandler();
    const processingOverlay = document.getElementById('processingOverlay');

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

    // Load model
    try {
        processingOverlay.style.display = 'flex';
        processingOverlay.textContent = 'Loading model...';
        await modelHandler.loadModel();
        processingOverlay.style.display = 'none';
    } catch (error) {
        showError('Failed to load model. Please refresh the page.');
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
        
        try {
            await classify();
        } catch (error) {
            showError('Classification failed. Please try again.');
        }
    });

    // Helper functions
    async function processImage(file) {
        setProcessing(true);
        try {
            await imageProcessor.processUploadedImage(file);
        } finally {
            setProcessing(false);
        }
    }

    async function startWebcam() {
        setProcessing(true);
        try {
            return await imageProcessor.setupWebcam(webcamElement);
        } finally {
            setProcessing(false);
        }
    }

    function stopWebcam() {
        imageProcessor.stopWebcam();
        isWebcamActive = false;
        webcamElement.style.display = 'none';
        webcamBtn.textContent = 'Use Webcam';
        classifyBtn.disabled = true;
    }

    async function classify() {
        setProcessing(true);
        processingOverlay.textContent = 'Classifying...';

        try {
            let imageData;
            if (isWebcamActive) {
                imageData = imageProcessor.captureWebcamImage();
            } else {
                imageData = imageProcessor.getCanvas();
            }

            const predictions = await modelHandler.predict(imageData);
            displayPredictions(predictions);
        } finally {
            setProcessing(false);
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

    function setProcessing(processing) {
        isProcessing = processing;
        processingOverlay.style.display = processing ? 'flex' : 'none';
        classifyBtn.disabled = processing;
    }

    function showError(message) {
        console.error(message);
        alert(message);
        setProcessing(false);
    }
}); 