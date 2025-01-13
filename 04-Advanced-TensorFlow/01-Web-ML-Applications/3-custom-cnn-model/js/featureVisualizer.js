class FeatureVisualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.colormap = this.generateColormap();
    }

    generateColormap() {
        // Generate viridis-like colormap
        const colors = [];
        for (let i = 0; i < 256; i++) {
            const t = i / 255;
            colors.push([
                this.interpolate(68, 33, 62, t),    // R
                this.interpolate(1, 144, 247, t),   // G
                this.interpolate(84, 212, 157, t)   // B
            ]);
        }
        return colors;
    }

    interpolate(start, mid, end, t) {
        if (t < 0.5) {
            return Math.round(start + (mid - start) * (t * 2));
        } else {
            return Math.round(mid + (end - mid) * ((t - 0.5) * 2));
        }
    }

    visualizeFeatureMap(featureMap) {
        const { data, width, height } = featureMap;
        
        // Set canvas size
        this.canvas.width = CONFIG.visualization.featureMapSize;
        this.canvas.height = CONFIG.visualization.featureMapSize;
        
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Find data range for normalization
        const min = Math.min(...data);
        const max = Math.max(...data);
        const range = max - min;
        
        // Calculate pixel size
        const pixelWidth = this.canvas.width / width;
        const pixelHeight = this.canvas.height / height;
        
        // Draw feature map
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const value = data[y * width + x];
                const normalizedValue = (value - min) / range;
                const colorIndex = Math.floor(normalizedValue * 255);
                const [r, g, b] = this.colormap[colorIndex];
                
                this.ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${CONFIG.visualization.alpha})`;
                this.ctx.fillRect(
                    x * pixelWidth,
                    y * pixelHeight,
                    pixelWidth,
                    pixelHeight
                );
            }
        }

        // Add grid lines for better visualization
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        this.ctx.lineWidth = 0.5;

        for (let x = 0; x <= width; x++) {
            this.ctx.beginPath();
            this.ctx.moveTo(x * pixelWidth, 0);
            this.ctx.lineTo(x * pixelWidth, this.canvas.height);
            this.ctx.stroke();
        }

        for (let y = 0; y <= height; y++) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y * pixelHeight);
            this.ctx.lineTo(this.canvas.width, y * pixelHeight);
            this.ctx.stroke();
        }
    }

    clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
} 