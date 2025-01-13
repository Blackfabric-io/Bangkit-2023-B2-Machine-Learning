const CONFIG = {
    model: {
        mobilenet: {
            version: 1,
            alpha: 1.0,
            inputShape: [150, 150],
            outputStride: 16
        },
        custom: {
            path: {
                json: 'mobilenet-model.json',
                weights: [
                    'group1-shard1of10.bin',
                    'group1-shard2of10.bin',
                    'group1-shard3of10.bin',
                    'group1-shard4of10.bin',
                    'group1-shard5of10.bin',
                    'group1-shard6of10.bin',
                    'group1-shard7of10.bin',
                    'group1-shard8of10.bin',
                    'group1-shard9of10.bin',
                    'group1-shard10of10.bin'
                ]
            }
        },
        featureLayers: {
            'conv_pw_13_relu': {
                description: 'High-level features',
                channels: 1024
            },
            'conv_pw_12_relu': {
                description: 'Mid-level features',
                channels: 512
            },
            'conv_pw_11_relu': {
                description: 'Low-level features',
                channels: 256
            }
        }
    },
    image: {
        maxSize: 150,
        normalize: true,
        meanRGB: [123.68, 116.779, 103.939],
        cropSize: 0.8
    },
    prediction: {
        topK: 5,
        confidenceThreshold: 0.1,
        batchSize: 1,
        classNames: {
            0: 'Cat',
            1: 'Dog',
            2: 'Bird',
            3: 'Fish',
            4: 'Horse'
        }
    },
    visualization: {
        featureMapSize: 150,
        maxFeatureMaps: 9,
        colormap: 'viridis',
        alpha: 0.7
    },
    performance: {
        maxMemoryMB: 512,
        targetMs: 100,
        warmupRuns: 3
    }
}; 