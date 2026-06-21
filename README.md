# Digital Media Analytics with Intelligent Decision Support System

A university-level data science and machine learning project that monitors three YouTube channels (Aaj TV, Hum TV, Raftar) to deliver end-to-end analytics on audience engagement, content performance, temporal trends, NLP-driven topic/sentiment analysis, graph-based relationship modeling, and evidence-based strategic decision support. Built as a DAV course project.

## Contributors

- [@talal](https://github.com/talal-11)
- [@ahmed](https://github.com/Ah65med)
- [@neha](https://github.com/nehalq)
- [@shayan](https://github.com/sfxdeve)

## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [YouTube API Key Setup](#youtube-api-key-setup)
- [Usage](#usage)
- [Jupyter Notebook](#jupyter-notebook)
- [Module Descriptions](#module-descriptions)
- [Output](#output)
- [Requirements](#requirements)
- [Limitations](#limitations)
- [License](#license)

## Project Structure

```
digital-media-analytics/
├── __init__.py                     # Package init with CONFIG dict and API key placeholder
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── src/
│   ├── __init__.py
│   ├── data_collection.py          # YouTube Data API v3 collector + synthetic fallback
│   ├── preprocessing.py            # Numerical, temporal, categorical, and text preprocessing
│   ├── nlp_pipeline.py             # Sentiment, embeddings, topic modeling, topic drift
│   ├── feature_engineering.py      # Unified feature matrix construction + selection
│   ├── modeling.py                 # Regression, classification, clustering, model persistence
│   ├── graph_analysis.py           # Heterogeneous graph, centrality, community detection
│   ├── visualization.py            # Publication-quality static + interactive visualizations
│   ├── decision_support.py         # Evidence-based strategic recommendation engine
│   └── synthetic_generation.py     # Privacy-preserving synthetic data generation
├── data/
│   ├── raw/                        # Raw collected data (Parquet)
│   ├── processed/                  # Preprocessed & feature-engineered data (Parquet)
│   └── synthetic/                  # Synthetic data outputs (Parquet)
├── notebooks/
│   └── digital_media_analytics.ipynb  # End-to-end Jupyter notebook (16 analysis sections)
└── outputs/
    ├── figures/                    # Generated plots and visualizations (PNG/SVG)
    ├── graphs/                     # Exported NetworkX graphs (GraphML/JSON)
    ├── models/                     # Serialized ML models (joblib)
    ├── reports/                    # Decision support & executive summary reports (JSON/MD/TXT)
    └── streamlit_dashboard.py      # Interactive Streamlit analytics dashboard
```

## Features

- **Data Collection** — Pulls channel stats, video metadata, comments, and replies via YouTube Data API v3; automatically falls back to synthetic data generation when no API key is provided.
- **Preprocessing** — Numerical scaling, temporal feature extraction, categorical encoding, text cleaning, outlier detection (Isolation Forest), and schema validation for all datasets.
- **NLP Pipeline** — VADER and Transformer-based sentiment analysis, Sentence-Transformer contextual embeddings, UMAP+HDBSCAN topic modeling, temporal topic drift detection, and discourse-level sentiment reporting.
- **Feature Engineering** — Constructs a unified feature matrix from video metadata, comment statistics, NLP outputs, and embedding vectors; includes feature selection, interaction terms, and dimensionality reduction.
- **ML Modeling** — Engagement regression (Linear/ML), content success classification (Logistic/RF/XGBoost), unsupervised clustering (K-Means/Agglomerative/HDBSCAN), temporal trend modeling, model interpretability (SHAP), and model persistence.
- **Graph Analytics** — Builds a heterogeneous NetworkX graph (channels, videos, comments, users, topics); community detection (Louvain), centrality metrics, link prediction, graph embeddings, and explainable analytics.
- **Visualization** — Statistical distributions, temporal trends, sentiment heatmaps, NLP topic landscapes, model performance dashboards, clustering projections, interactive Plotly charts, and a standalone Streamlit dashboard.
- **Decision Support** — Generates actionable strategic recommendations across six dimensions: posting strategy, content optimization, audience engagement, topic strategy, risk alerts, and channel comparison; exports structured reports (JSON, Markdown, plain text).

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd digital-media-analytics

# Install Python dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm

# Download NLTK data
python -m nltk.downloader wordnet stopwords punkt vader_lexicon
```

## YouTube API Key Setup

This project uses the YouTube Data API v3 to collect real channel data. Without a key, the system falls back to synthetic data generation automatically.

### Getting a Key

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project (or select an existing one).
3. Navigate to **APIs & Services > Library** and enable the **YouTube Data API v3**.
4. Go to **APIs & Services > Credentials**, click **Create Credentials > API Key**.
5. Copy the generated key.

### Providing Your Key

**Option A — Edit the CONFIG dict** in `__init__.py`:

```python
CONFIG = {
    ...
    "YOUTUBE_API_KEY": "YOUR_YOUTUBE_API_KEY_HERE",  # replace with your key
    ...
}
```

**Option B — Set an environment variable**:

```bash
export YOUTUBE_API_KEY="your-key-here"
```

The data collector checks `os.environ.get("YOUTUBE_API_KEY")` first, then falls back to the `CONFIG` dict value.

**No key?** The system automatically switches to `SyntheticDataCollector`, which generates realistic synthetic data for all three channels using statistical distribution fitting and text augmentation. A warning is printed to indicate synthetic mode.

## Usage

### Quick Start

```bash
# Step 1: Collect data (real or synthetic)
python src/data_collection.py

# Step 2: Run preprocessing
python src/preprocessing.py

# Step 3: Run NLP pipeline
python src/nlp_pipeline.py

# Step 4: Run feature engineering
python src/feature_engineering.py

# Step 5: Run ML modeling
python src/modeling.py

# Step 6: Run graph analysis
python src/graph_analysis.py

# Step 7: Generate visualizations
python src/visualization.py

# Step 8: Generate decision support reports
python src/decision_support.py
```

Alternatively, run all the above sections in order via the Jupyter notebook (see below).

### Full Pipeline (Placeholder)

A unified pipeline runner is planned. When available, run:

```bash
python -m digital_media_analytics.src.main
```

### Synthetic Data Generation (Standalone)

To generate synthetic data from existing processed data (e.g., for privacy-preserving sharing):

```bash
python src/synthetic_generation.py
```

## Jupyter Notebook

The notebook `notebooks/digital_media_analytics.ipynb` walks through the entire project in **16 analysis sections**:

1. Environment Setup & Configuration
2. Data Collection (YouTube API / Synthetic)
3. Exploratory Data Analysis
4. Numerical & Temporal Preprocessing
5. Text Preprocessing & Cleaning
6. Sentiment Analysis (VADER + Transformers)
7. Contextual Embeddings & Topic Modeling
8. Topic Drift & Temporal NLP Trends
9. Feature Engineering & Selection
10. Regression: Engagement Prediction
11. Classification: Content Success Categorization
12. Unsupervised Clustering
13. Graph Construction & Network Analytics
14. Community Detection & Link Prediction
15. Visualization Suite
16. Decision Support & Strategic Recommendations

## Module Descriptions

| Module | Description |
|---|---|
| `data_collection.py` | Collects channel, video, comment, and reply data from YouTube API v3 with automatic synthetic fallback |
| `preprocessing.py` | Advanced preprocessing pipeline covering numerical, temporal, categorical, and text data with schema validation |
| `nlp_pipeline.py` | End-to-end NLP: sentiment analysis, contextual embeddings, topic modeling, and temporal topic drift detection |
| `feature_engineering.py` | Builds a unified feature matrix combining metadata, NLP outputs, embeddings, and interaction features |
| `modeling.py` | ML pipeline: engagement regression, content success classification, clustering, and model persistence |
| `graph_analysis.py` | Heterogeneous graph construction with centrality analysis, community detection (Louvain), and link prediction |
| `visualization.py` | Publication-quality visualizations using Matplotlib, Seaborn, and Plotly; exports to PNG, SVG, and HTML |
| `decision_support.py` | Evidence-based recommendation engine generating structured strategic reports across six business dimensions |
| `synthetic_generation.py` | Privacy-preserving synthetic data generation via statistical fitting, correlation preservation, and text augmentation |

## Output

After running the pipeline, the following outputs are generated:

| Directory | Contents |
|---|---|
| `data/raw/` | Raw channel stats, video metadata, comments, and replies (Parquet) |
| `data/processed/` | Preprocessed DataFrames with engineered features (Parquet) |
| `data/synthetic/` | Synthetic versions of all datasets for privacy-preserving sharing (Parquet) |
| `outputs/figures/` | Statistical plots, temporal trends, sentiment distributions, topic landscapes, model diagnostics (PNG) |
| `outputs/graphs/` | Exported NetworkX graph objects and visualizations (GraphML, JSON, PNG) |
| `outputs/models/` | Serialized ML models — regressors, classifiers, clusterers (joblib) |
| `outputs/reports/` | Decision support report (JSON, Markdown, TXT) and executive summary (TXT) |

## Requirements

- **Python** 3.9 or later
- **Operating System**: macOS, Linux, or Windows (WSL recommended)
- **Memory**: 8 GB RAM minimum (16 GB recommended for full Transformer-based NLP and graph embeddings)
- **Disk**: ~500 MB for dependencies; ~2–5 GB for data, models, and outputs depending on collection volume
- **Internet**: Required for YouTube API data collection, first-time model downloads, and NLTK/spaCy resource downloads

All Python dependencies are listed in `requirements.txt` and include pandas, numpy, scikit-learn, matplotlib, seaborn, plotly, networkx, nltk, spacy, transformers, sentence-transformers, torch, xgboost, statsmodels, pyvis, python-louvain, umap-learn, hdbscan, google-api-python-client, datasets, streamlit, shap, dash, pyarrow, and others.

## Limitations

- **YouTube API Quotas** — The YouTube Data API v3 imposes a daily quota (default 10,000 units). Full data collection across three channels can exhaust this quickly. The synthetic fallback mode bypasses this entirely.
- **Synthetic Data Caveats** — Synthetic data preserves statistical properties (distribution shape, correlations) but does not reflect real-world events, trending topics, or actual audience behavior. It is suitable for pipeline testing and methodology demonstration but not for production insights.
- **Model Compute Requirements** — Transformer-based sentiment analysis (`cardiffnlp/twitter-roberta-base-sentiment-latest`) and embedding models (`all-MiniLM-L6-v2`) require GPU acceleration for reasonable throughput on large datasets. CPU-only execution is supported but significantly slower.
- **Memory Usage** — The full pipeline loads multiple DataFrames and model artifacts simultaneously. On systems with less than 8 GB RAM, consider reducing `MAX_VIDEOS_PER_CHANNEL` and `MAX_COMMENTS_PER_VIDEO` in the CONFIG dict.
- **Notebook Module Imports** — When running from the Jupyter notebook, ensure the project root directory is on `sys.path` so that `from src import ...` imports resolve correctly.
