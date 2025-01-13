class ImageProcessor {
    constructor(previewCanvas) {
        this.canvas = previewCanvas;
        this.ctx = previewCanvas.getContext('2d');
        this.webcam = null;
        this.setupCanvas();
    }

    setupCanvas() {
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
                URL.revokeObjectURL(img.src);
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

        // Center the image in the canvas
        this.canvas.width = maxSize;
        this.canvas.height = maxSize;

        // Fill background
        this.ctx.fillStyle = '#f0f0f0';
        this.ctx.fillRect(0, 0, maxSize, maxSize);

        // Draw image centered
        const x = (maxSize - width) / 2;
        const y = (maxSize - height) / 2;
        
        this.ctx.drawImage(img, x, y, width, height);
    }

    async setupWebcam(videoElement) {
        try {
            const constraints = {
                video: {
                    facingMode: 'environment',
                    width: { ideal: CONFIG.image.maxSize },
                    height: { ideal: CONFIG.image.maxSize }
                }
            };

            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            videoElement.srcObject = stream;
            this.webcam = videoElement;

            // Wait for video to be ready
            await new Promise(resolve => {
                videoElement.onloadedmetadata = resolve;
            });

            // Set canvas size to match video
            this.canvas.width = CONFIG.image.maxSize;
            this.canvas.height = CONFIG.image.maxSize;

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

        // Calculate dimensions to maintain aspect ratio and center
        const videoAspect = this.webcam.videoWidth / this.webcam.videoHeight;
        let drawWidth = CONFIG.image.maxSize;
        let drawHeight = CONFIG.image.maxSize;
        let x = 0;
        let y = 0;

        if (videoAspect > 1) {
            drawHeight = drawWidth / videoAspect;
            y = (CONFIG.image.maxSize - drawHeight) / 2;
        } else {
            drawWidth = drawHeight * videoAspect;
            x = (CONFIG.image.maxSize - drawWidth) / 2;
        }

        // Clear and draw
        this.ctx.fillStyle = '#f0f0f0';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(this.webcam, x, y, drawWidth, drawHeight);

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

    clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
} 