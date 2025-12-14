const models = [
    {
        "id": "nomic-ai-modernbert",
        "name": "Nomic Ai Modernbert",
        "org": "Community",
        "modelId": "nomic-ai-modernbert",
        "params": "TODO",
        "maxDim": 768,
        "dimensions": {
            "float": [
                768
            ],
            "bfloat16": [
                768
            ],
            "binary": [
                768
            ]
        },
        "speeds": {
            "t4": 0,
            "c7g": 0
        },
        "mrlSupport": false,
        "binarySupport": true,
        "bfloat16Support": true,
        "scores": {
            "semantic_768_float": 0.612,
            "fusion_768_float": 0.6,
            "atan_norm_768_float": 0.63,
            "norm_linear_768_float": 0.626,
            "semantic_768_bfloat16": 0.612,
            "fusion_768_bfloat16": 0.6,
            "atan_norm_768_bfloat16": 0.63,
            "norm_linear_768_bfloat16": 0.626,
            "semantic_768_binary": 0.582,
            "fusion_768_binary": 0.59,
            "atan_norm_768_binary": 0.557,
            "norm_linear_768_binary": 0.619
        }
    },
    {
        "id": "gte-multilingual-base",
        "name": "GTE Multilingual Base",
        "org": "Community",
        "modelId": "gte-multilingual-base",
        "params": "TODO",
        "maxDim": 768,
        "dimensions": {
            "float": [
                768
            ],
            "bfloat16": [
                768
            ],
            "binary": [
                768
            ]
        },
        "speeds": {
            "t4": 0,
            "c7g": 0
        },
        "mrlSupport": false,
        "binarySupport": true,
        "bfloat16Support": true,
        "scores": {
            "semantic_768_float": 0.609,
            "fusion_768_float": 0.614,
            "atan_norm_768_float": 0.641,
            "norm_linear_768_float": 0.646,
            "semantic_768_bfloat16": 0.608,
            "fusion_768_bfloat16": 0.614,
            "atan_norm_768_bfloat16": 0.642,
            "norm_linear_768_bfloat16": 0.646,
            "semantic_768_binary": 0.0,
            "fusion_768_binary": 0.0,
            "atan_norm_768_binary": 0.0,
            "norm_linear_768_binary": 0.0
        }
    },
    {
        "id": "lightonai-modernbert-large",
        "name": "Lightonai Modernbert Large",
        "org": "Community",
        "modelId": "lightonai-modernbert-large",
        "params": "TODO",
        "maxDim": 1024,
        "dimensions": {
            "float": [
                1024
            ],
            "bfloat16": [
                1024
            ],
            "binary": [
                1024
            ]
        },
        "speeds": {
            "t4": 0,
            "c7g": 0
        },
        "mrlSupport": false,
        "binarySupport": true,
        "bfloat16Support": true,
        "scores": {
            "semantic_1024_float": 0.62,
            "fusion_1024_float": 0.608,
            "atan_norm_1024_float": 0.637,
            "norm_linear_1024_float": 0.636,
            "semantic_1024_bfloat16": 0.619,
            "fusion_1024_bfloat16": 0.608,
            "atan_norm_1024_bfloat16": 0.637,
            "norm_linear_1024_bfloat16": 0.636,
            "semantic_1024_binary": 0.593,
            "fusion_1024_binary": 0.603,
            "atan_norm_1024_binary": 0.55,
            "norm_linear_1024_binary": 0.632
        }
    },
    {
        "id": "embeddinggemma-300m",
        "name": "Embeddinggemma 300M",
        "org": "Community",
        "modelId": "embeddinggemma-300m",
        "params": "TODO",
        "maxDim": 768,
        "dimensions": {
            "float": [
                768,
                384
            ],
            "bfloat16": [
                768,
                384
            ],
            "binary": [
                768,
                384
            ]
        },
        "speeds": {
            "t4": 0,
            "c7g": 0
        },
        "mrlSupport": true,
        "binarySupport": true,
        "bfloat16Support": true,
        "scores": {
            "semantic_768_float": 0.631,
            "fusion_768_float": 0.613,
            "atan_norm_768_float": 0.638,
            "norm_linear_768_float": 0.638,
            "semantic_384_float": 0.607,
            "fusion_384_float": 0.609,
            "atan_norm_384_float": 0.637,
            "norm_linear_384_float": 0.635,
            "semantic_768_bfloat16": 0.631,
            "fusion_768_bfloat16": 0.614,
            "atan_norm_768_bfloat16": 0.637,
            "norm_linear_768_bfloat16": 0.638,
            "semantic_384_bfloat16": 0.607,
            "fusion_384_bfloat16": 0.609,
            "atan_norm_384_bfloat16": 0.637,
            "norm_linear_384_bfloat16": 0.635,
            "semantic_768_binary": 0.591,
            "fusion_768_binary": 0.605,
            "atan_norm_768_binary": 0.553,
            "norm_linear_768_binary": 0.622,
            "semantic_384_binary": 0.535,
            "fusion_384_binary": 0.582,
            "atan_norm_384_binary": 0.563,
            "norm_linear_384_binary": 0.604
        }
    },
    {
        "id": "e5-small-v2",
        "name": "E5 Small V2",
        "org": "Community",
        "modelId": "e5-small-v2",
        "params": "TODO",
        "maxDim": 384,
        "dimensions": {
            "float": [
                384
            ],
            "bfloat16": [
                384
            ],
            "binary": [
                384
            ]
        },
        "speeds": {
            "t4": 0,
            "c7g": 0
        },
        "mrlSupport": false,
        "binarySupport": true,
        "bfloat16Support": true,
        "scores": {
            "semantic_384_float": 0.566,
            "fusion_384_float": 0.598,
            "atan_norm_384_float": 0.617,
            "norm_linear_384_float": 0.619,
            "semantic_384_bfloat16": 0.567,
            "fusion_384_bfloat16": 0.598,
            "atan_norm_384_bfloat16": 0.617,
            "norm_linear_384_bfloat16": 0.619,
            "semantic_384_binary": 0.311,
            "fusion_384_binary": 0.472,
            "atan_norm_384_binary": 0.557,
            "norm_linear_384_binary": 0.508
        }
    },
    {
        "id": "multilingual-e5-base",
        "name": "Multilingual E5 Base",
        "org": "Community",
        "modelId": "multilingual-e5-base",
        "params": "TODO",
        "maxDim": 768,
        "dimensions": {
            "float": [
                768
            ],
            "bfloat16": [
                768
            ],
            "binary": [
                768
            ]
        },
        "speeds": {
            "t4": 0,
            "c7g": 0
        },
        "mrlSupport": false,
        "binarySupport": true,
        "bfloat16Support": true,
        "scores": {
            "semantic_768_float": 0.58,
            "fusion_768_float": 0.601,
            "atan_norm_768_float": 0.621,
            "norm_linear_768_float": 0.624,
            "semantic_768_bfloat16": 0.579,
            "fusion_768_bfloat16": 0.6,
            "atan_norm_768_bfloat16": 0.621,
            "norm_linear_768_bfloat16": 0.624,
            "semantic_768_binary": 0.37,
            "fusion_768_binary": 0.522,
            "atan_norm_768_binary": 0.557,
            "norm_linear_768_binary": 0.546
        }
    },
    {
        "id": "alibaba-gte-modernbert",
        "name": "Alibaba GTE Modernbert",
        "org": "Community",
        "modelId": "alibaba-gte-modernbert",
        "params": "TODO",
        "maxDim": 768,
        "dimensions": {
            "float": [
                768
            ],
            "bfloat16": [
                768
            ],
            "binary": [
                768
            ]
        },
        "speeds": {
            "t4": 0,
            "c7g": 0
        },
        "mrlSupport": false,
        "binarySupport": true,
        "bfloat16Support": true,
        "scores": {
            "semantic_768_float": 0.641,
            "fusion_768_float": 0.621,
            "atan_norm_768_float": 0.666,
            "norm_linear_768_float": 0.663,
            "semantic_768_bfloat16": 0.642,
            "fusion_768_bfloat16": 0.621,
            "atan_norm_768_bfloat16": 0.666,
            "norm_linear_768_bfloat16": 0.663,
            "semantic_768_binary": 0.625,
            "fusion_768_binary": 0.62,
            "atan_norm_768_binary": 0.575,
            "norm_linear_768_binary": 0.652
        }
    },
    {
        "id": "bm25",
        "name": "BM25",
        "org": "Vespa",
        "modelId": "vespa-bm25",
        "params": "N/A",
        "maxDim": null,
        "speeds": {
            "t4": 0,
            "c7g": 0
        },
        "mrlSupport": false,
        "binarySupport": false,
        "bfloat16Support": false,
        "isBM25": true,
        "scores": {
            "semantic_float": 0.527,
            "fusion_float": null,
            "semantic_bfloat16": null,
            "fusion_bfloat16": null,
            "semantic_binary": null,
            "fusion_binary": null
        }
    }
];