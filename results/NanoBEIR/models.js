const models = [
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
            "c7g": 12.08
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
        },
        "benchmarks": {
            "g4dn.xlarge": {
                "hardware_type": "g4dn.xlarge",
                "model_id": "lightonai-modernbert-large",
                "hf_repo": "lightonai/modernbert-embed-large",
                "model_url": "https://huggingface.co/lightonai/modernbert-embed-large/resolve/main/onnx/model.onnx",
                "commit_sha": "95a19bff4963",
                "model_size_mb": 1506.38,
                "embedding_dim": 1024,
                "queries_samples_processed": 951,
                "queries_avg_latency_ms": 10.14,
                "queries_p95_latency_ms": 10.23,
                "queries_throughput": 95.01,
                "docs_samples_processed": 274,
                "docs_avg_latency_ms": 35.3,
                "docs_p95_latency_ms": 35.91,
                "docs_throughput": 27.39
            },
            "c7g.2xlarge": {
                "hardware_type": "c7g.2xlarge",
                "model_id": "lightonai-modernbert-large",
                "hf_repo": "lightonai/modernbert-embed-large",
                "model_url": "https://huggingface.co/lightonai/modernbert-embed-large/resolve/main/onnx/model.onnx",
                "commit_sha": "95a19bff4963",
                "model_size_mb": 1506.38,
                "embedding_dim": 1024,
                "queries_samples_processed": 121,
                "queries_avg_latency_ms": 82.24,
                "queries_p95_latency_ms": 82.71,
                "queries_throughput": 12.08,
                "docs_samples_processed": 17,
                "docs_avg_latency_ms": 614.76,
                "docs_p95_latency_ms": 616.24,
                "docs_throughput": 1.62
            },
            "m8g.xlarge": {
                "hardware_type": "m8g.xlarge",
                "model_id": "lightonai-modernbert-large",
                "hf_repo": "lightonai/modernbert-embed-large",
                "model_url": "https://huggingface.co/lightonai/modernbert-embed-large/resolve/main/onnx/model.onnx",
                "commit_sha": "95a19bff4963",
                "model_size_mb": 1506.38,
                "embedding_dim": 1024,
                "queries_samples_processed": 122,
                "queries_avg_latency_ms": 81.34,
                "queries_p95_latency_ms": 88.41,
                "queries_throughput": 12.2,
                "docs_samples_processed": 19,
                "docs_avg_latency_ms": 536.47,
                "docs_p95_latency_ms": 577.68,
                "docs_throughput": 1.86
            }
        }
    },
    {
        "id": "e5-base-v2",
        "name": "E5 Base V2",
        "org": "Community",
        "modelId": "e5-base-v2",
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
            "c7g": 53.8
        },
        "mrlSupport": false,
        "binarySupport": true,
        "bfloat16Support": true,
        "scores": {
            "semantic_768_float": 0.593,
            "fusion_768_float": 0.597,
            "atan_norm_768_float": 0.628,
            "norm_linear_768_float": 0.632,
            "semantic_768_bfloat16": 0.593,
            "fusion_768_bfloat16": 0.597,
            "atan_norm_768_bfloat16": 0.628,
            "norm_linear_768_bfloat16": 0.632,
            "semantic_768_binary": 0.423,
            "fusion_768_binary": 0.544,
            "atan_norm_768_binary": 0.563,
            "norm_linear_768_binary": 0.566
        },
        "benchmarks": {
            "g4dn.xlarge": {
                "hardware_type": "g4dn.xlarge",
                "model_id": "e5-base-v2",
                "hf_repo": "intfloat/e5-base-v2",
                "model_url": "https://huggingface.co/intfloat/e5-base-v2/resolve/main/onnx/model.onnx",
                "commit_sha": "f52bf8ec8c71",
                "model_size_mb": 415.62,
                "embedding_dim": 768,
                "queries_samples_processed": 3214,
                "queries_avg_latency_ms": 2.9,
                "queries_p95_latency_ms": 2.92,
                "queries_throughput": 321.39,
                "docs_samples_processed": 1232,
                "docs_avg_latency_ms": 7.45,
                "docs_p95_latency_ms": 7.52,
                "docs_throughput": 123.11
            },
            "c7g.2xlarge": {
                "hardware_type": "c7g.2xlarge",
                "model_id": "e5-base-v2",
                "hf_repo": "intfloat/e5-base-v2",
                "model_url": "https://huggingface.co/intfloat/e5-base-v2/resolve/main/onnx/model.onnx",
                "commit_sha": "f52bf8ec8c71",
                "model_size_mb": 415.62,
                "embedding_dim": 768,
                "queries_samples_processed": 539,
                "queries_avg_latency_ms": 18.24,
                "queries_p95_latency_ms": 18.4,
                "queries_throughput": 53.8,
                "docs_samples_processed": 73,
                "docs_avg_latency_ms": 136.27,
                "docs_p95_latency_ms": 136.37,
                "docs_throughput": 7.29
            },
            "m8g.xlarge": {
                "hardware_type": "m8g.xlarge",
                "model_id": "e5-base-v2",
                "hf_repo": "intfloat/e5-base-v2",
                "model_url": "https://huggingface.co/intfloat/e5-base-v2/resolve/main/onnx/model.onnx",
                "commit_sha": "f52bf8ec8c71",
                "model_size_mb": 415.62,
                "embedding_dim": 768,
                "queries_samples_processed": 555,
                "queries_avg_latency_ms": 17.65,
                "queries_p95_latency_ms": 18.68,
                "queries_throughput": 55.48,
                "docs_samples_processed": 83,
                "docs_avg_latency_ms": 120.55,
                "docs_p95_latency_ms": 139.72,
                "docs_throughput": 8.22
            }
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
        },
        "benchmarks": {}
    },
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
            "c7g": 34.64
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
            "semantic_768_binary": 0.584,
            "fusion_768_binary": 0.59,
            "atan_norm_768_binary": 0.557,
            "norm_linear_768_binary": 0.619
        },
        "benchmarks": {
            "g4dn.xlarge": {
                "hardware_type": "g4dn.xlarge",
                "model_id": "nomic-ai-modernbert",
                "hf_repo": "nomic-ai/modernbert-embed-base",
                "model_url": "https://huggingface.co/nomic-ai/modernbert-embed-base/resolve/main/onnx/model.onnx",
                "commit_sha": "d556a88e3325",
                "model_size_mb": 568.86,
                "embedding_dim": 768,
                "queries_samples_processed": 1763,
                "queries_avg_latency_ms": 5.41,
                "queries_p95_latency_ms": 5.58,
                "queries_throughput": 176.28,
                "docs_samples_processed": 707,
                "docs_avg_latency_ms": 13.45,
                "docs_p95_latency_ms": 13.64,
                "docs_throughput": 70.65
            },
            "c7g.2xlarge": {
                "hardware_type": "c7g.2xlarge",
                "model_id": "nomic-ai-modernbert",
                "hf_repo": "nomic-ai/modernbert-embed-base",
                "model_url": "https://huggingface.co/nomic-ai/modernbert-embed-base/resolve/main/onnx/model.onnx",
                "commit_sha": "d556a88e3325",
                "model_size_mb": 568.86,
                "embedding_dim": 768,
                "queries_samples_processed": 347,
                "queries_avg_latency_ms": 28.44,
                "queries_p95_latency_ms": 28.6,
                "queries_throughput": 34.64,
                "docs_samples_processed": 49,
                "docs_avg_latency_ms": 206.32,
                "docs_p95_latency_ms": 206.94,
                "docs_throughput": 4.82
            },
            "m8g.xlarge": {
                "hardware_type": "m8g.xlarge",
                "model_id": "nomic-ai-modernbert",
                "hf_repo": "nomic-ai/modernbert-embed-base",
                "model_url": "https://huggingface.co/nomic-ai/modernbert-embed-base/resolve/main/onnx/model.onnx",
                "commit_sha": "d556a88e3325",
                "model_size_mb": 568.86,
                "embedding_dim": 768,
                "queries_samples_processed": 339,
                "queries_avg_latency_ms": 29.03,
                "queries_p95_latency_ms": 29.76,
                "queries_throughput": 33.86,
                "docs_samples_processed": 54,
                "docs_avg_latency_ms": 184.66,
                "docs_p95_latency_ms": 203.77,
                "docs_throughput": 5.38
            }
        }
    },
    {
        "id": "alibaba-gte-modernbert-int8",
        "name": "Alibaba GTE Modernbert Int8",
        "org": "Community",
        "modelId": "alibaba-gte-modernbert-int8",
        "params": "TODO",
        "maxDim": 768,
        "dimensions": {
            "float": [],
            "bfloat16": [],
            "binary": [
                768
            ]
        },
        "speeds": {
            "t4": 0,
            "c7g": 108.56
        },
        "mrlSupport": false,
        "binarySupport": true,
        "bfloat16Support": false,
        "scores": {
            "semantic_768_binary": 0.605,
            "fusion_768_binary": 0.625,
            "atan_norm_768_binary": 0.647,
            "norm_linear_768_binary": 0.654
        },
        "benchmarks": {
            "g4dn.xlarge": {
                "hardware_type": "g4dn.xlarge",
                "model_id": "alibaba-gte-modernbert-int8",
                "hf_repo": "Alibaba-NLP/gte-modernbert-base",
                "model_url": "https://huggingface.co/Alibaba-NLP/gte-modernbert-base/resolve/main/onnx/model_int8.onnx",
                "commit_sha": "e7f32e3c00f9",
                "model_size_mb": 143.26,
                "embedding_dim": 768,
                "queries_samples_processed": 433,
                "queries_avg_latency_ms": 22.61,
                "queries_p95_latency_ms": 25.24,
                "queries_throughput": 43.27,
                "docs_samples_processed": 107,
                "docs_avg_latency_ms": 92.25,
                "docs_p95_latency_ms": 105.85,
                "docs_throughput": 10.68
            },
            "c7g.2xlarge": {
                "hardware_type": "c7g.2xlarge",
                "model_id": "alibaba-gte-modernbert-int8",
                "hf_repo": "Alibaba-NLP/gte-modernbert-base",
                "model_url": "https://huggingface.co/Alibaba-NLP/gte-modernbert-base/resolve/main/onnx/model_int8.onnx",
                "commit_sha": "e7f32e3c00f9",
                "model_size_mb": 143.26,
                "embedding_dim": 768,
                "queries_samples_processed": 1086,
                "queries_avg_latency_ms": 8.82,
                "queries_p95_latency_ms": 8.92,
                "queries_throughput": 108.56,
                "docs_samples_processed": 170,
                "docs_avg_latency_ms": 58.07,
                "docs_p95_latency_ms": 58.21,
                "docs_throughput": 16.95
            },
            "m8g.xlarge": {
                "hardware_type": "m8g.xlarge",
                "model_id": "alibaba-gte-modernbert-int8",
                "hf_repo": "Alibaba-NLP/gte-modernbert-base",
                "model_url": "https://huggingface.co/Alibaba-NLP/gte-modernbert-base/resolve/main/onnx/model_int8.onnx",
                "commit_sha": "e7f32e3c00f9",
                "model_size_mb": 143.26,
                "embedding_dim": 768,
                "queries_samples_processed": 955,
                "queries_avg_latency_ms": 9.99,
                "queries_p95_latency_ms": 11.01,
                "queries_throughput": 95.4,
                "docs_samples_processed": 162,
                "docs_avg_latency_ms": 60.83,
                "docs_p95_latency_ms": 63.96,
                "docs_throughput": 16.16
            }
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
            "c7g": 371.6
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
        },
        "benchmarks": {
            "g4dn.xlarge": {
                "hardware_type": "g4dn.xlarge",
                "model_id": "e5-small-v2",
                "hf_repo": "Xenova/e5-small-v2",
                "model_url": "https://huggingface.co/Xenova/e5-small-v2/resolve/main/onnx/model_int8.onnx",
                "commit_sha": "02af79985278",
                "model_size_mb": 32.2,
                "embedding_dim": 384,
                "queries_samples_processed": 1254,
                "queries_avg_latency_ms": 7.56,
                "queries_p95_latency_ms": 8.21,
                "queries_throughput": 125.37,
                "docs_samples_processed": 378,
                "docs_avg_latency_ms": 25.23,
                "docs_p95_latency_ms": 26.09,
                "docs_throughput": 37.78
            },
            "c7g.2xlarge": {
                "hardware_type": "c7g.2xlarge",
                "model_id": "e5-small-v2",
                "hf_repo": "Xenova/e5-small-v2",
                "model_url": "https://huggingface.co/Xenova/e5-small-v2/resolve/main/onnx/model_int8.onnx",
                "commit_sha": "02af79985278",
                "model_size_mb": 32.2,
                "embedding_dim": 384,
                "queries_samples_processed": 3717,
                "queries_avg_latency_ms": 2.48,
                "queries_p95_latency_ms": 2.53,
                "queries_throughput": 371.6,
                "docs_samples_processed": 663,
                "docs_avg_latency_ms": 14.39,
                "docs_p95_latency_ms": 14.49,
                "docs_throughput": 66.23
            },
            "m8g.xlarge": {
                "hardware_type": "m8g.xlarge",
                "model_id": "e5-small-v2",
                "hf_repo": "Xenova/e5-small-v2",
                "model_url": "https://huggingface.co/Xenova/e5-small-v2/resolve/main/onnx/model_int8.onnx",
                "commit_sha": "02af79985278",
                "model_size_mb": 32.2,
                "embedding_dim": 384,
                "queries_samples_processed": 3301,
                "queries_avg_latency_ms": 2.7,
                "queries_p95_latency_ms": 3.05,
                "queries_throughput": 329.99,
                "docs_samples_processed": 667,
                "docs_avg_latency_ms": 14.23,
                "docs_p95_latency_ms": 14.92,
                "docs_throughput": 66.64
            }
        }
    },
    {
        "id": "embeddinggemma-300m-q4",
        "name": "Embeddinggemma 300M Q4",
        "org": "Community",
        "modelId": "embeddinggemma-300m-q4",
        "params": "TODO",
        "maxDim": 768,
        "dimensions": {
            "float": [
                768,
                512,
                128
            ],
            "bfloat16": [
                768,
                512,
                128
            ],
            "binary": [
                768,
                512,
                128
            ]
        },
        "speeds": {
            "t4": 0,
            "c7g": 39.16
        },
        "mrlSupport": true,
        "binarySupport": true,
        "bfloat16Support": true,
        "scores": {
            "semantic_768_float": 0.622,
            "fusion_768_float": 0.605,
            "atan_norm_768_float": 0.636,
            "norm_linear_768_float": 0.635,
            "semantic_512_float": 0.619,
            "fusion_512_float": 0.612,
            "atan_norm_512_float": 0.638,
            "norm_linear_512_float": 0.638,
            "semantic_128_float": 0.461,
            "fusion_128_float": 0.547,
            "atan_norm_128_float": 0.568,
            "norm_linear_128_float": 0.578,
            "semantic_768_bfloat16": 0.622,
            "fusion_768_bfloat16": 0.605,
            "atan_norm_768_bfloat16": 0.636,
            "norm_linear_768_bfloat16": 0.635,
            "semantic_512_bfloat16": 0.619,
            "fusion_512_bfloat16": 0.612,
            "atan_norm_512_bfloat16": 0.639,
            "norm_linear_512_bfloat16": 0.638,
            "semantic_128_bfloat16": 0.461,
            "fusion_128_bfloat16": 0.549,
            "atan_norm_128_bfloat16": 0.568,
            "norm_linear_128_bfloat16": 0.578,
            "semantic_768_binary": 0.578,
            "fusion_768_binary": 0.598,
            "atan_norm_768_binary": 0.547,
            "norm_linear_768_binary": 0.618,
            "semantic_512_binary": 0.548,
            "fusion_512_binary": 0.589,
            "atan_norm_512_binary": 0.554,
            "norm_linear_512_binary": 0.609,
            "semantic_128_binary": 0.322,
            "fusion_128_binary": 0.472,
            "atan_norm_128_binary": 0.565,
            "norm_linear_128_binary": 0.51
        },
        "benchmarks": {
            "c7g.2xlarge": {
                "hardware_type": "c7g.2xlarge",
                "model_id": "embeddinggemma-300m-q4",
                "hf_repo": "onnx-community/embeddinggemma-300m-ONNX",
                "model_url": "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/onnx/model_q4.onnx",
                "commit_sha": "5090578d9565",
                "model_size_mb": 188.11,
                "embedding_dim": 768,
                "queries_samples_processed": 392,
                "queries_avg_latency_ms": 25.1,
                "queries_p95_latency_ms": 25.36,
                "queries_throughput": 39.16,
                "docs_samples_processed": 59,
                "docs_avg_latency_ms": 168.74,
                "docs_p95_latency_ms": 168.91,
                "docs_throughput": 5.89
            },
            "m8g.xlarge": {
                "hardware_type": "m8g.xlarge",
                "model_id": "embeddinggemma-300m-q4",
                "hf_repo": "onnx-community/embeddinggemma-300m-ONNX",
                "model_url": "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/onnx/model_q4.onnx",
                "commit_sha": "5090578d9565",
                "model_size_mb": 188.11,
                "embedding_dim": 768,
                "queries_samples_processed": 410,
                "queries_avg_latency_ms": 23.96,
                "queries_p95_latency_ms": 24.9,
                "queries_throughput": 40.93,
                "docs_samples_processed": 63,
                "docs_avg_latency_ms": 159.47,
                "docs_p95_latency_ms": 177.69,
                "docs_throughput": 6.23
            }
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
            "c7g": 43.11
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
        },
        "benchmarks": {
            "g4dn.xlarge": {
                "hardware_type": "g4dn.xlarge",
                "model_id": "alibaba-gte-modernbert",
                "hf_repo": "Alibaba-NLP/gte-modernbert-base",
                "model_url": "https://huggingface.co/Alibaba-NLP/gte-modernbert-base/resolve/main/onnx/model.onnx",
                "commit_sha": "e7f32e3c00f9",
                "model_size_mb": 568.76,
                "embedding_dim": 768,
                "queries_samples_processed": 1927,
                "queries_avg_latency_ms": 4.99,
                "queries_p95_latency_ms": 5.11,
                "queries_throughput": 192.62,
                "docs_samples_processed": 745,
                "docs_avg_latency_ms": 12.81,
                "docs_p95_latency_ms": 13.0,
                "docs_throughput": 74.5
            },
            "c7g.2xlarge": {
                "hardware_type": "c7g.2xlarge",
                "model_id": "alibaba-gte-modernbert",
                "hf_repo": "Alibaba-NLP/gte-modernbert-base",
                "model_url": "https://huggingface.co/Alibaba-NLP/gte-modernbert-base/resolve/main/onnx/model.onnx",
                "commit_sha": "e7f32e3c00f9",
                "model_size_mb": 568.76,
                "embedding_dim": 768,
                "queries_samples_processed": 432,
                "queries_avg_latency_ms": 22.79,
                "queries_p95_latency_ms": 22.9,
                "queries_throughput": 43.11,
                "docs_samples_processed": 50,
                "docs_avg_latency_ms": 200.33,
                "docs_p95_latency_ms": 202.0,
                "docs_throughput": 4.97
            },
            "m8g.xlarge": {
                "hardware_type": "m8g.xlarge",
                "model_id": "alibaba-gte-modernbert",
                "hf_repo": "Alibaba-NLP/gte-modernbert-base",
                "model_url": "https://huggingface.co/Alibaba-NLP/gte-modernbert-base/resolve/main/onnx/model.onnx",
                "commit_sha": "e7f32e3c00f9",
                "model_size_mb": 568.76,
                "embedding_dim": 768,
                "queries_samples_processed": 413,
                "queries_avg_latency_ms": 23.8,
                "queries_p95_latency_ms": 25.95,
                "queries_throughput": 41.2,
                "docs_samples_processed": 56,
                "docs_avg_latency_ms": 179.39,
                "docs_p95_latency_ms": 201.12,
                "docs_throughput": 5.54
            }
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
            "c7g": 47.02
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
        },
        "benchmarks": {
            "g4dn.xlarge": {
                "hardware_type": "g4dn.xlarge",
                "model_id": "multilingual-e5-base",
                "hf_repo": "intfloat/multilingual-e5-base",
                "model_url": "https://huggingface.co/intfloat/multilingual-e5-base/resolve/main/onnx/model.onnx",
                "commit_sha": "835193815a39",
                "model_size_mb": 1058.63,
                "embedding_dim": 768,
                "queries_samples_processed": 3164,
                "queries_avg_latency_ms": 2.95,
                "queries_p95_latency_ms": 2.98,
                "queries_throughput": 316.35,
                "docs_samples_processed": 1037,
                "docs_avg_latency_ms": 8.98,
                "docs_p95_latency_ms": 9.14,
                "docs_throughput": 103.67
            },
            "c7g.2xlarge": {
                "hardware_type": "c7g.2xlarge",
                "model_id": "multilingual-e5-base",
                "hf_repo": "intfloat/multilingual-e5-base",
                "model_url": "https://huggingface.co/intfloat/multilingual-e5-base/resolve/main/onnx/model.onnx",
                "commit_sha": "835193815a39",
                "model_size_mb": 1058.63,
                "embedding_dim": 768,
                "queries_samples_processed": 471,
                "queries_avg_latency_ms": 20.92,
                "queries_p95_latency_ms": 21.05,
                "queries_throughput": 47.02,
                "docs_samples_processed": 63,
                "docs_avg_latency_ms": 158.93,
                "docs_p95_latency_ms": 159.12,
                "docs_throughput": 6.25
            },
            "m8g.xlarge": {
                "hardware_type": "m8g.xlarge",
                "model_id": "multilingual-e5-base",
                "hf_repo": "intfloat/multilingual-e5-base",
                "model_url": "https://huggingface.co/intfloat/multilingual-e5-base/resolve/main/onnx/model.onnx",
                "commit_sha": "835193815a39",
                "model_size_mb": 1058.63,
                "embedding_dim": 768,
                "queries_samples_processed": 467,
                "queries_avg_latency_ms": 21.04,
                "queries_p95_latency_ms": 21.82,
                "queries_throughput": 46.65,
                "docs_samples_processed": 71,
                "docs_avg_latency_ms": 140.92,
                "docs_p95_latency_ms": 160.86,
                "docs_throughput": 7.04
            }
        }
    },
    {
        "id": "snowflake-arctic-embed-m-v2.0-int8",
        "name": "Snowflake Arctic Embed M V2.0 Int8",
        "org": "Community",
        "modelId": "snowflake-arctic-embed-m-v2.0-int8",
        "params": "TODO",
        "maxDim": 768,
        "dimensions": {
            "float": [],
            "bfloat16": [
                768
            ],
            "binary": [
                768
            ]
        },
        "speeds": {
            "t4": 0,
            "c7g": 111.76
        },
        "mrlSupport": false,
        "binarySupport": true,
        "bfloat16Support": true,
        "scores": {
            "semantic_768_bfloat16": 0.652,
            "fusion_768_bfloat16": 0.634,
            "atan_norm_768_bfloat16": 0.666,
            "norm_linear_768_bfloat16": 0.663,
            "semantic_768_binary": 0.614,
            "fusion_768_binary": 0.617,
            "atan_norm_768_binary": 0.647,
            "norm_linear_768_binary": 0.646
        },
        "benchmarks": {
            "g4dn.xlarge": {
                "hardware_type": "g4dn.xlarge",
                "model_id": "snowflake-arctic-embed-m-v2.0-int8",
                "hf_repo": "Snowflake/snowflake-arctic-embed-m-v2.0",
                "model_url": "https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0/resolve/main/onnx/model_int8.onnx",
                "commit_sha": "95c274148085",
                "model_size_mb": 296.51,
                "embedding_dim": 768,
                "queries_samples_processed": 532,
                "queries_avg_latency_ms": 18.28,
                "queries_p95_latency_ms": 23.32,
                "queries_throughput": 53.17,
                "docs_samples_processed": 121,
                "docs_avg_latency_ms": 81.25,
                "docs_p95_latency_ms": 103.52,
                "docs_throughput": 12.1
            },
            "c7g.2xlarge": {
                "hardware_type": "c7g.2xlarge",
                "model_id": "snowflake-arctic-embed-m-v2.0-int8",
                "hf_repo": "Snowflake/snowflake-arctic-embed-m-v2.0",
                "model_url": "https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0/resolve/main/onnx/model_int8.onnx",
                "commit_sha": "95c274148085",
                "model_size_mb": 296.51,
                "embedding_dim": 768,
                "queries_samples_processed": 1118,
                "queries_avg_latency_ms": 8.61,
                "queries_p95_latency_ms": 8.7,
                "queries_throughput": 111.76,
                "docs_samples_processed": 191,
                "docs_avg_latency_ms": 51.59,
                "docs_p95_latency_ms": 51.73,
                "docs_throughput": 19.05
            },
            "m8g.xlarge": {
                "hardware_type": "m8g.xlarge",
                "model_id": "snowflake-arctic-embed-m-v2.0-int8",
                "hf_repo": "Snowflake/snowflake-arctic-embed-m-v2.0",
                "model_url": "https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0/resolve/main/onnx/model_int8.onnx",
                "commit_sha": "95c274148085",
                "model_size_mb": 296.51,
                "embedding_dim": 768,
                "queries_samples_processed": 972,
                "queries_avg_latency_ms": 9.92,
                "queries_p95_latency_ms": 10.64,
                "queries_throughput": 97.01,
                "docs_samples_processed": 185,
                "docs_avg_latency_ms": 53.4,
                "docs_p95_latency_ms": 56.47,
                "docs_throughput": 18.4
            }
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
            "c7g": 30.73
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
        },
        "benchmarks": {
            "g4dn.xlarge": {
                "hardware_type": "g4dn.xlarge",
                "model_id": "embeddinggemma-300m",
                "hf_repo": "onnx-community/embeddinggemma-300m-ONNX",
                "model_url": "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/onnx/model_fp16.onnx",
                "commit_sha": "5090578d9565",
                "model_size_mb": 589.46,
                "embedding_dim": 768,
                "queries_samples_processed": 373,
                "queries_avg_latency_ms": 26.33,
                "queries_p95_latency_ms": 28.27,
                "queries_throughput": 37.26,
                "docs_samples_processed": 128,
                "docs_avg_latency_ms": 77.33,
                "docs_p95_latency_ms": 78.93,
                "docs_throughput": 12.78
            },
            "c7g.2xlarge": {
                "hardware_type": "c7g.2xlarge",
                "model_id": "embeddinggemma-300m",
                "hf_repo": "onnx-community/embeddinggemma-300m-ONNX",
                "model_url": "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/onnx/model_fp16.onnx",
                "commit_sha": "5090578d9565",
                "model_size_mb": 589.46,
                "embedding_dim": 768,
                "queries_samples_processed": 308,
                "queries_avg_latency_ms": 32.01,
                "queries_p95_latency_ms": 32.16,
                "queries_throughput": 30.73,
                "docs_samples_processed": 52,
                "docs_avg_latency_ms": 193.98,
                "docs_p95_latency_ms": 194.21,
                "docs_throughput": 5.13
            },
            "m8g.xlarge": {
                "hardware_type": "m8g.xlarge",
                "model_id": "embeddinggemma-300m",
                "hf_repo": "onnx-community/embeddinggemma-300m-ONNX",
                "model_url": "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/onnx/model_fp16.onnx",
                "commit_sha": "5090578d9565",
                "model_size_mb": 589.46,
                "embedding_dim": 768,
                "queries_samples_processed": 306,
                "queries_avg_latency_ms": 32.11,
                "queries_p95_latency_ms": 35.38,
                "queries_throughput": 30.53,
                "docs_samples_processed": 57,
                "docs_avg_latency_ms": 175.0,
                "docs_p95_latency_ms": 196.77,
                "docs_throughput": 5.67
            }
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