"""Digital Media Analytics with Intelligent Decision Support System.

A complete Python-based data science and machine learning project that
monitors and analyzes multiple YouTube channels, generating insights about
audience engagement, content performance, temporal trends, discussion
patterns, topic evolution, graph-based relationships, and strategic
decision-making.
"""

__version__ = "1.0.0"
__author__ = "Digital Media Analytics Project"

CONFIG = {
    "CHANNELS": [
        {
            "name": "Aaj TV (Aaj News)",
            "channel_id": "UCgBAPAcLsh_MAPvJprIz89w",
        },
        {
            "name": "Hum TV",
            "channel_id": "UCEeEQxm6qc_qaTE7qTV5aLQ",
        },
        {
            "name": "Raftar",
            "channel_id": "UC6zIImBjDqtEsVZfQLPoQSw",
        },
    ],
    "DATA_DIR": "./data",
    "RAW_DATA_DIR": "./data/raw",
    "PROCESSED_DATA_DIR": "./data/processed",
    "SYNTHETIC_DATA_DIR": "./data/synthetic",
    "OUTPUT_DIR": "./outputs",
    "FIGURES_DIR": "./outputs/figures",
    "GRAPHS_DIR": "./outputs/graphs",
    "MODELS_DIR": "./outputs/models",
    "REPORTS_DIR": "./outputs/reports",
    "BATCH_SIZE": 512,
    "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "RANDOM_STATE": 42,
    # YouTube Data API key placeholder — replace with your own key
    "YOUTUBE_API_KEY": "YOUR_YOUTUBE_API_KEY_HERE",
    "YOUTUBE_API_SERVICE_NAME": "youtube",
    "YOUTUBE_API_VERSION": "v3",
    # Synthetic data parameters
    "SYNTHETIC_MULTIPLIER": 3,
    "NOISE_STD": 0.1,
    "SEMANTIC_SIMILARITY_THRESHOLD": 0.75,
    "MAX_VIDEOS_PER_CHANNEL": 200,
    "MAX_COMMENTS_PER_VIDEO": 100,
    # NLP parameters
    "SENTIMENT_MODEL": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "SPACY_MODEL": "en_core_web_sm",
    # Modeling parameters
    "TEST_SIZE": 0.2,
    "CV_FOLDS": 5,
    # Success classification thresholds
    "SUCCESS_HIGH_PERCENTILE": 75,
    "SUCCESS_LOW_PERCENTILE": 25,
    # Chunked processing
    "CHUNK_SIZE": 5000,
    "MAX_MEMORY_GB": 4,
}

import os

for key in [
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "SYNTHETIC_DATA_DIR",
    "OUTPUT_DIR",
    "FIGURES_DIR",
    "GRAPHS_DIR",
    "MODELS_DIR",
    "REPORTS_DIR",
]:
    os.makedirs(CONFIG[key], exist_ok=True)
