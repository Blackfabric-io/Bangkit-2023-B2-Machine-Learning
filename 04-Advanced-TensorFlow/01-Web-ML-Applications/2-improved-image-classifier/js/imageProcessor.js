class ImageProcessor {
    constructor(previewCanvas) {
        this.canvas = previewCanvas;
        this.ctx = previewCanvas.getContext('2d');
        this.webcam = null;
        this.setupCanvas();
    }

    setupCanvas() {
        // Enable image smoothing for better quality
        this.ctx.imageSmoothingEnabled = true;
        this.ctx.imageSmoothingQuality = 'high';
    }

    async processUploadedImage(file) {
        return new Promise((resolve, reject) => {
            if (!file.type.startsWith('image/')) {
                reject(new Error('Please upload an image file'));
                return;
            }

            const img = new Image();
            img.onload = () => {
                this.drawImage(img);
                URL.revokeObjectURL(img.src); // Clean up
                resolve(this.canvas);
            };
            img.onerror = () => {
                reject(new Error('Failed to load image'));
                URL.revokeObjectURL(img.src);
            };
            img.src = URL.createObjectURL(file);
        });
    }

    drawImage(img) {
        // Calculate dimensions maintaining aspect ratio
        const maxSize = CONFIG.image.maxSize;
        let width = img.width;
        let height = img.height;
        
        if (width > height) {
            if (width > maxSize) {
                height = height * (maxSize / width);
                width = maxSize;
            }
        } else {
            if (height > maxSize) {
                width = width * (maxSize / height);
                height = maxSize;
            }
        }

        // Set canvas size
        this.canvas.width = width;
        this.canvas.height = height;

        // Clear and draw
        this.ctx.clearRect(0, 0, width, height);
        this.ctx.drawImage(img, 0, 0, width, height);

        // Convert to grayscale if needed
        if (CONFIG.image.grayscale) {
            this.convertToGrayscale();
        }
    }

    convertToGrayscale() {
        const imageData = this.ctx.getImageData(
            0, 0, 
            this.canvas.width, 
            this.canvas.height
        );
        const data = imageData.data;

        for (let i = 0; i < data.length; i += 4) {
            const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
            data[i] = avg;     // R
            data[i + 1] = avg; // G
            data[i + 2] = avg; // B
        }

        this.ctx.putImageData(imageData, 0, 0);
    }

    async setupWebcam(videoElement) {
        try {
            const constraints = {
                video: {
                    facingMode: 'environment',
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                }
            };

            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            videoElement.srcObject = stream;
            this.webcam = videoElement;

            // Wait for video to be ready
            await new Promise(resolve => {
                videoElement.onloadedmetadata = resolve;
            });

            return true;
        } catch (error) {
            console.error('Error accessing webcam:', error);
            return false;
        }
    }

    captureWebcamImage() {
        if (!this.webcam || !this.webcam.videoWidth) {
            throw new Error('Webcam not initialized or ready');
        }

        // Set canvas dimensions to match video
        this.canvas.width = this.webcam.videoWidth;
        this.canvas.height = this.webcam.videoHeight;

        // Draw current video frame
        this.ctx.drawImage(this.webcam, 0, 0);

        // Convert to grayscale if needed
        if (CONFIG.image.grayscale) {
            this.convertToGrayscale();
        }

        return this.canvas;
    }

    stopWebcam() {
        if (this.webcam && this.webcam.srcObject) {
            const tracks = this.webcam.srcObject.getTracks();
            tracks.forEach(track => track.stop());
            this.webcam.srcObject = null;
        }
    }

    getCanvas() {
        return this.canvas;
    }
} 