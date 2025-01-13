class ImageProcessor {
    constructor(previewCanvas) {
        this.canvas = previewCanvas;
        this.ctx = previewCanvas.getContext('2d');
        this.webcam = null;
    }

    async processUploadedImage(file) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                this.drawImage(img);
                resolve(this.canvas);
            };
            img.onerror = reject;
            img.src = URL.createObjectURL(file);
        });
    }

    drawImage(img) {
        // Set canvas size to maintain aspect ratio
        const aspectRatio = img.width / img.height;
        this.canvas.width = 400;
        this.canvas.height = 400 / aspectRatio;

        // Clear canvas and draw image
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
    }

    async setupWebcam(videoElement) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'environment'
                }
            });
            videoElement.srcObject = stream;
            this.webcam = videoElement;
            return true;
        } catch (error) {
            console.error('Error accessing webcam:', error);
            return false;
        }
    }

    captureWebcamImage() {
        if (!this.webcam) {
            throw new Error('Webcam not initialized');
        }

        // Set canvas dimensions to match video
        this.canvas.width = this.webcam.videoWidth;
        this.canvas.height = this.webcam.videoHeight;

        // Draw current video frame
        this.ctx.drawImage(this.webcam, 0, 0);
        return this.canvas;
    }

    stopWebcam() {
        if (this.webcam && this.webcam.srcObject) {
            const tracks = this.webcam.srcObject.getTracks();
            tracks.forEach(track => track.stop());
            this.webcam.srcObject = null;
        }
    }
} 