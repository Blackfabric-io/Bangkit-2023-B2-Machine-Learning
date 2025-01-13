class PerformanceMonitor {
    constructor() {
        this.metrics = {
            inferenceTime: [],
            memoryUsage: [],
            fps: []
        };
        this.lastFrameTime = 0;
        this.frameCount = 0;
        this.maxSamples = 100;
    }

    recordMetrics(stats) {
        // Record inference time
        this.metrics.inferenceTime.push(stats.inferenceTime);
        if (this.metrics.inferenceTime.length > this.maxSamples) {
            this.metrics.inferenceTime.shift();
        }

        // Record memory usage
        this.metrics.memoryUsage.push(stats.memoryUsage);
        if (this.metrics.memoryUsage.length > this.maxSamples) {
            this.metrics.memoryUsage.shift();
        }

        // Calculate FPS
        const now = performance.now();
        if (this.lastFrameTime) {
            const fps = 1000 / (now - this.lastFrameTime);
            this.metrics.fps.push(fps);
            if (this.metrics.fps.length > this.maxSamples) {
                this.metrics.fps.shift();
            }
        }
        this.lastFrameTime = now;

        // Update UI
        this.updateMetricsDisplay();
    }

    updateMetricsDisplay() {
        // Update inference time display
        const avgInferenceTime = this.calculateAverage(this.metrics.inferenceTime);
        const inferenceTimeElement = document.getElementById('inferenceTime');
        if (inferenceTimeElement) {
            inferenceTimeElement.textContent = `${avgInferenceTime.toFixed(1)} ms`;
        }

        // Update memory usage display
        const avgMemoryUsage = this.calculateAverage(this.metrics.memoryUsage);
        const memoryUsageElement = document.getElementById('memoryUsage');
        if (memoryUsageElement) {
            memoryUsageElement.textContent = `${avgMemoryUsage.toFixed(1)} MB`;
        }

        // Check performance thresholds
        this.checkPerformanceThresholds(avgInferenceTime, avgMemoryUsage);
    }

    calculateAverage(array) {
        if (array.length === 0) return 0;
        const sum = array.reduce((a, b) => a + b, 0);
        return sum / array.length;
    }

    checkPerformanceThresholds(inferenceTime, memoryUsage) {
        // Check inference time threshold
        if (inferenceTime > CONFIG.performance.targetMs) {
            console.warn(`Inference time (${inferenceTime.toFixed(1)}ms) exceeds target (${CONFIG.performance.targetMs}ms)`);
        }

        // Check memory usage threshold
        if (memoryUsage > CONFIG.performance.maxMemoryMB) {
            console.warn(`Memory usage (${memoryUsage.toFixed(1)}MB) exceeds limit (${CONFIG.performance.maxMemoryMB}MB)`);
        }
    }

    getMetricsSummary() {
        return {
            inferenceTime: {
                avg: this.calculateAverage(this.metrics.inferenceTime),
                min: Math.min(...this.metrics.inferenceTime),
                max: Math.max(...this.metrics.inferenceTime)
            },
            memoryUsage: {
                avg: this.calculateAverage(this.metrics.memoryUsage),
                min: Math.min(...this.metrics.memoryUsage),
                max: Math.max(...this.metrics.memoryUsage)
            },
            fps: {
                avg: this.calculateAverage(this.metrics.fps),
                min: Math.min(...this.metrics.fps),
                max: Math.max(...this.metrics.fps)
            }
        };
    }

    reset() {
        this.metrics = {
            inferenceTime: [],
            memoryUsage: [],
            fps: []
        };
        this.lastFrameTime = 0;
        this.frameCount = 0;
    }
} 