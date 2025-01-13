document.addEventListener('DOMContentLoaded', async () => {
    // Initialize components
    const imagePreview = document.getElementById('imagePreview');
    const imageProcessor = new ImageProcessor(imagePreview);
    const modelHandler = new ModelHandler();

    // UI Elements
    const uploadBtn = document.getElementById('uploadBtn');
    const imageUpload = document.getElementById('imageUpload');
    const webcamBtn = document.getElementById('webcamBtn');
    const webcamElement = document.getElementById('webcam');
    const classifyBtn = document.getElementById('classifyBtn');
    const predictionsDiv = document.getElementById('predictions');

    // Load model
    await modelHandler.loadModel();

    // Event Listeners
    uploadBtn.addEventListener('click', () => {
        imageUpload.click();
    });

    imageUpload.addEventListener('change', async (e) => {
        if (e.target.files.length > 0) {
            await imageProcessor.processUploadedImage(e.target.files[0]);
            classifyBtn.disabled = false;
            webcamElement.style.display = 'none';
        }
    });

    let isWebcamActive = false;
    webcamBtn.addEventListener('click', async () => {
        if (!isWebcamActive) {
            const success = await imageProcessor.setupWebcam(webcamElement);
            if (success) {
                isWebcamActive = true;
                webcamElement.style.display = 'block';
                webcamBtn.textContent = 'Stop Webcam';
                classifyBtn.disabled = false;
            }
        } else {
            imageProcessor.stopWebcam();
            isWebcamActive = false;
            webcamElement.style.display = 'none';
            webcamBtn.textContent = 'Use Webcam';
            classifyBtn.disabled = true;
        }
    });

    classifyBtn.addEventListener('click', async () => {
        try {
            let imageData;
            if (isWebcamActive) {
                imageData = imageProcessor.captureWebcamImage();
            } else {
                imageData = imagePreview;
            }

            const predictions = await modelHandler.predict(imageData);
            displayPredictions(predictions);
        } catch (error) {
            console.error('Classification error:', error);
            alert('Error during classification. Please try again.');
        }
    });

    function displayPredictions(predictions) {
        predictionsDiv.innerHTML = '';
        predictions.forEach(pred => {
            const predictionItem = document.createElement('div');
            predictionItem.className = 'prediction-item';
            
            const confidenceBar = document.createElement('div');
            confidenceBar.className = 'confidence-bar';
            confidenceBar.style.width = `${pred.confidence}%`;

            predictionItem.innerHTML = `
                <span>${pred.class}</span>
                <span>${pred.confidence}%</span>
            `;
            predictionItem.appendChild(confidenceBar);
            predictionsDiv.appendChild(predictionItem);
        });
    }
}); 