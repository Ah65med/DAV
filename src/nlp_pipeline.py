"""
NLP Pipeline for Digital Media Analytics.

Provides sentiment analysis (VADER primary, Transformers fallback),
contextual embeddings (Sentence-Transformers primary, TF-IDF+PCA fallback),
topic modeling (UMAP+HDBSCAN primary, PCA+KMeans fallback), temporal topic
drift detection, and discourse-level sentiment reporting.

All heavy models are loaded lazily on first use to reduce memory footprint.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("nlp_pipeline")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG: Dict[str, Any] = {
    "BATCH_SIZE": 512,
    "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "SENTIMENT_MODEL": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "SPACY_MODEL": "en_core_web_sm",
    "RANDOM_STATE": 42,
    "PROCESSED_DATA_DIR": "./data/processed",
}

# ---------------------------------------------------------------------------
# NLPPipeline
# ---------------------------------------------------------------------------


class NLPPipeline:
    """End-to-end NLP pipeline for digital media comment & video analysis.

    Parameters
    ----------
    batch_size : int
        Default batch size for processing (default 512).
    embedding_model : str or None
        HuggingFace model name for sentence-transformers. When *None* the
        value from ``CONFIG["EMBEDDING_MODEL"]`` is used.
    device : str or None
        PyTorch device string (e.g. ``"cpu"``, ``"cuda"``). Defaults to
        ``"cpu"`` to avoid unexpected GPU requirements.
    """

    def __init__(
        self,
        batch_size: int = 512,
        embedding_model: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.batch_size = batch_size
        self.embedding_model_name = embedding_model or CONFIG["EMBEDDING_MODEL"]
        self.device = device or "cpu"

        # Lazily-loaded internals
        self._vader: Optional[Any] = None
        self._sentiment_pipeline: Optional[Any] = None
        self._sentiment_pipeline_failed: bool = False
        self._embedding_model: Optional[Any] = None
        self._embedding_model_failed: bool = False

    # ------------------------------------------------------------------
    # Sentiment helpers
    # ------------------------------------------------------------------

    def _load_sentiment_model(self) -> None:
        """Lazy-load VADER sentiment analyser and optionally a transformers pipeline."""
        if self._vader is not None:
            return

        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            self._vader = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyser loaded.")
        except ImportError as exc:
            raise ImportError(
                "vaderSentiment is required for sentiment analysis. "
                "Install it with: pip install vaderSentiment"
            ) from exc

        # Try loading the transformers sentiment model as secondary
        if self._sentiment_pipeline is not None or self._sentiment_pipeline_failed:
            return

        try:
            from transformers import pipeline

            self._sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=CONFIG["SENTIMENT_MODEL"],
                device=self.device if self.device != "cpu" else -1,
            )
            logger.info("Transformers sentiment pipeline loaded.")
        except ImportError:
            self._sentiment_pipeline_failed = True
            logger.warning(
                "transformers not installed — sentiment will use VADER only. "
                "Install with: pip install transformers"
            )
        except Exception:
            self._sentiment_pipeline_failed = True
            logger.warning(
                "Could not load transformers sentiment model. Falling back to VADER only."
            )

    def _vader_scores_to_df(self, scores: List[Dict[str, float]]) -> pd.DataFrame:
        """Convert raw VADER scores into a labelled DataFrame."""
        records = []
        for s in scores:
            compound = s["compound"]
            if compound >= 0.05:
                label = "positive"
            elif compound <= -0.05:
                label = "negative"
            else:
                label = "neutral"
            records.append(
                {
                    "sentiment_score": compound,
                    "sentiment_label": label,
                    "sentiment_pos": s["pos"],
                    "sentiment_neu": s["neu"],
                    "sentiment_neg": s["neg"],
                }
            )
        return pd.DataFrame(records)

    def analyze_sentiment_vader(self, texts: Sequence[str]) -> pd.DataFrame:
        """Run VADER sentiment on a sequence of texts.

        Returns a DataFrame with columns: *sentiment_score*,
        *sentiment_label*, *sentiment_pos*, *sentiment_neu*, *sentiment_neg*.
        """
        self._load_sentiment_model()
        results = []
        for text in texts:
            try:
                results.append(self._vader.polarity_scores(str(text or "")))
            except Exception:
                results.append({"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0})
        return self._vader_scores_to_df(results)

    def _transformers_label_to_uniform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Map transformers output dict to a uniform schema."""
        label = (result.get("label") or "").lower()
        score = result.get("score", 0.0)

        if label == "positive":
            compound, pos, neu, neg, lbl = score, score, 0.0, 0.0, "positive"
        elif label == "negative":
            compound, pos, neu, neg, lbl = -score, 0.0, 0.0, score, "negative"
        else:
            compound, pos, neu, neg, lbl = 0.0, 0.0, 1.0, 0.0, "neutral"

        return {
            "sentiment_score": compound,
            "sentiment_label": lbl,
            "sentiment_pos": pos,
            "sentiment_neu": neu,
            "sentiment_neg": neg,
        }

    def analyze_sentiment_transformers(self, texts: Sequence[str]) -> pd.DataFrame:
        """Run transformers sentiment pipeline, falling back to VADER on failure."""
        if self._sentiment_pipeline is None and not self._sentiment_pipeline_failed:
            self._load_sentiment_model()

        if self._sentiment_pipeline is not None:
            try:
                results = self._sentiment_pipeline(
                    [str(t or "") for t in texts],
                    batch_size=self.batch_size,
                    truncation=True,
                )
                return pd.DataFrame(
                    [self._transformers_label_to_uniform(r) for r in results]
                )
            except Exception:
                logger.warning(
                    "Transformers sentiment failed on current batch; falling back to VADER."
                )

        logger.info("Using VADER as fallback for transformers sentiment.")
        return self.analyze_sentiment_vader(texts)

    def analyze_sentiment(
        self,
        texts: Sequence[str],
        method: str = "vader",
    ) -> pd.DataFrame:
        """Main sentiment entry-point. Processes *texts* in batches.

        Parameters
        ----------
        texts : sequence of str
        method : {"vader", "transformers"}
            Which sentiment engine to prefer.

        Returns
        -------
        pd.DataFrame
        """
        texts = list(texts)
        method_func = (
            self.analyze_sentiment_transformers
            if method == "transformers"
            else self.analyze_sentiment_vader
        )

        frames: List[pd.DataFrame] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                frames.append(method_func(batch))
            except Exception:
                logger.exception("Sentiment batch %d failed; using neutral fallback.", i)
                frames.append(
                    pd.DataFrame(
                        [
                            {
                                "sentiment_score": 0.0,
                                "sentiment_label": "neutral",
                                "sentiment_pos": 0.0,
                                "sentiment_neu": 1.0,
                                "sentiment_neg": 0.0,
                            }
                        ]
                        * len(batch)
                    )
                )

        return (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(
                columns=[
                    "sentiment_score",
                    "sentiment_label",
                    "sentiment_pos",
                    "sentiment_neu",
                    "sentiment_neg",
                ]
            )
        )

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_columns(df: pd.DataFrame, required: List[str], context: str) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"{context}: missing columns {missing}")

    def aggregate_sentiment_by_video(self, comments_df: pd.DataFrame) -> pd.DataFrame:
        """Group by *video_id* and compute mean sentiment + std (variance proxy)."""
        self._check_columns(
            comments_df, ["video_id", "sentiment_score"], "aggregate_sentiment_by_video"
        )
        grouped = comments_df.groupby("video_id", as_index=False).agg(
            mean_sentiment=("sentiment_score", "mean"),
            sentiment_std=("sentiment_score", "std"),
            comment_count=("sentiment_score", "count"),
        )
        grouped["sentiment_std"] = grouped["sentiment_std"].fillna(0.0)
        return grouped

    def aggregate_sentiment_by_channel(
        self,
        videos_df: pd.DataFrame,
        comments_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Join videos & comments, then aggregate sentiment per channel."""
        self._check_columns(
            videos_df, ["video_id", "channel_id"], "aggregate_sentiment_by_channel:videos"
        )
        self._check_columns(
            comments_df,
            ["video_id", "sentiment_score"],
            "aggregate_sentiment_by_channel:comments",
        )

        merged = comments_df.merge(videos_df[["video_id", "channel_id"]], on="video_id")
        grouped = merged.groupby("channel_id", as_index=False).agg(
            mean_sentiment=("sentiment_score", "mean"),
            sentiment_std=("sentiment_score", "std"),
            comment_count=("sentiment_score", "count"),
        )
        grouped["sentiment_std"] = grouped["sentiment_std"].fillna(0.0)
        return grouped

    def aggregate_sentiment_by_time(
        self,
        comments_df: pd.DataFrame,
        window: str = "W",
    ) -> pd.DataFrame:
        """Group sentiment by time *window* (pandas frequency alias, default 'W' = weekly)."""
        self._check_columns(
            comments_df,
            ["published_at", "sentiment_score"],
            "aggregate_sentiment_by_time",
        )

        df = comments_df.copy()
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
        df["time_bucket"] = df["published_at"].dt.to_period(window).astype(str)

        grouped = df.groupby("time_bucket", as_index=False).agg(
            mean_sentiment=("sentiment_score", "mean"),
            sentiment_std=("sentiment_score", "std"),
            comment_count=("sentiment_score", "count"),
        )
        grouped["sentiment_std"] = grouped["sentiment_std"].fillna(0.0)
        return grouped.sort_values("time_bucket")

    def aggregate_sentiment_by_topic(
        self,
        comments_df: pd.DataFrame,
        topic_assignments: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute sentiment aggregates per topic.

        *topic_assignments* must contain at least *comment_id* and *topic_label*.
        """
        self._check_columns(
            topic_assignments, ["comment_id", "topic_label"], "aggregate_sentiment_by_topic"
        )
        self._check_columns(
            comments_df, ["comment_id", "sentiment_score"], "aggregate_sentiment_by_topic"
        )

        merged = comments_df.merge(topic_assignments, on="comment_id", how="inner")
        if merged.empty:
            return pd.DataFrame(
                columns=["topic_label", "mean_sentiment", "sentiment_std", "comment_count"]
            )

        grouped = merged.groupby("topic_label", as_index=False).agg(
            mean_sentiment=("sentiment_score", "mean"),
            sentiment_std=("sentiment_score", "std"),
            comment_count=("sentiment_score", "count"),
        )
        grouped["sentiment_std"] = grouped["sentiment_std"].fillna(0.0)
        return grouped

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def _load_embedding_model(self) -> None:
        """Lazy-load sentence-transformers model; mark as failed on error."""
        if self._embedding_model is not None or self._embedding_model_failed:
            return

        try:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(
                self.embedding_model_name, device=self.device
            )
            logger.info("Sentence-Transformer model '%s' loaded.", self.embedding_model_name)
        except ImportError:
            self._embedding_model_failed = True
            logger.warning(
                "sentence-transformers not installed. Embeddings will use TF-IDF + PCA. "
                "Install with: pip install sentence-transformers"
            )
        except Exception:
            self._embedding_model_failed = True
            logger.warning(
                "Could not load Sentence-Transformer '%s'. Using TF-IDF + PCA fallback.",
                self.embedding_model_name,
            )

    def _tfidf_pca_embeddings(self, texts: Sequence[str]) -> np.ndarray:
        """Fallback: compute TF-IDF vectors then reduce with PCA."""
        from sklearn.decomposition import PCA
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
        tfidf = vectorizer.fit_transform([str(t or "") for t in texts])

        n_components = min(128, tfidf.shape[0] - 1, tfidf.shape[1])
        if n_components < 1:
            n_components = 1

        pca = PCA(n_components=n_components, random_state=CONFIG["RANDOM_STATE"])
        dense = tfidf.toarray() if hasattr(tfidf, "toarray") else tfidf.A
        return pca.fit_transform(dense).astype(np.float32)

    def generate_embeddings(
        self,
        texts: Sequence[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Generate dense embeddings for *texts*, processing in batches.

        Falls back to TF-IDF + PCA when Sentence-Transformers is unavailable.
        """
        texts = [str(t or "") for t in texts]
        if not texts:
            return np.array([], dtype=np.float32)

        self._load_embedding_model()

        if self._embedding_model is None:
            logger.info("Sentence-Transformer unavailable; falling back to TF-IDF + PCA.")
            return self._tfidf_pca_embeddings(texts)

        embeddings: List[np.ndarray] = []
        total = len(texts)
        for i in range(0, total, self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                batch_emb = self._embedding_model.encode(
                    batch,
                    batch_size=self.batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True,
                )
                embeddings.append(batch_emb)
            except Exception:
                logger.exception("Embedding batch %d failed; using TF-IDF+PCA fallback.", i)
                return self._tfidf_pca_embeddings(texts)

        return np.vstack(embeddings) if embeddings else np.array([], dtype=np.float32)

    def generate_comment_embeddings(
        self,
        comments_df: pd.DataFrame,
        text_column: str = "comment_text",
    ) -> np.ndarray:
        """Generate embeddings for comment texts."""
        self._check_columns(comments_df, [text_column], "generate_comment_embeddings")
        return self.generate_embeddings(comments_df[text_column].tolist())

    def generate_title_embeddings(self, videos_df: pd.DataFrame) -> np.ndarray:
        """Generate embeddings for video titles."""
        self._check_columns(videos_df, ["title"], "generate_title_embeddings")
        return self.generate_embeddings(videos_df["title"].tolist())

    def generate_description_embeddings(self, videos_df: pd.DataFrame) -> np.ndarray:
        """Generate embeddings for video descriptions."""
        self._check_columns(videos_df, ["description"], "generate_description_embeddings")
        return self.generate_embeddings(videos_df["description"].fillna("").tolist())

    @staticmethod
    def average_embedding(embeddings_list: Sequence[np.ndarray]) -> np.ndarray:
        """Average a sequence of embedding vectors into a single vector."""
        if not embeddings_list:
            return np.array([], dtype=np.float32)
        return np.mean(np.stack(embeddings_list), axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Topic modeling
    # ------------------------------------------------------------------

    @staticmethod
    def _reduce_embeddings(
        embeddings: np.ndarray,
        n_components: int = 5,
    ) -> np.ndarray:
        """Dimensionality reduction: UMAP when available, else PCA."""
        try:
            import umap

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reducer = umap.UMAP(
                    n_components=n_components,
                    random_state=CONFIG["RANDOM_STATE"],
                    n_neighbors=15,
                    min_dist=0.0,
                    metric="cosine",
                )
            return reducer.fit_transform(embeddings)
        except ImportError:
            logger.warning("umap-learn not available; falling back to PCA.")
        except Exception:
            logger.warning("UMAP reduction failed; falling back to PCA.")

        from sklearn.decomposition import PCA

        n_comp = min(n_components, embeddings.shape[0] - 1, embeddings.shape[1])
        if n_comp < 1:
            n_comp = 1
        pca = PCA(n_components=n_comp, random_state=CONFIG["RANDOM_STATE"])
        return pca.fit_transform(embeddings)

    def _cluster_reduced(
        self,
        reduced: np.ndarray,
        min_cluster_size: int = 15,
    ) -> np.ndarray:
        """Cluster reduced embeddings: HDBSCAN when available, else KMeans."""
        try:
            import hdbscan

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=5,
                    metric="euclidean",
                    cluster_selection_epsilon=0.1,
                )
            return clusterer.fit_predict(reduced)
        except ImportError:
            logger.warning("hdbscan not available; falling back to KMeans.")
        except Exception:
            logger.warning("HDBSCAN failed; falling back to KMeans.")

        from sklearn.cluster import KMeans

        n_clusters = max(1, min(len(reduced) // min_cluster_size, 20))
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=CONFIG["RANDOM_STATE"],
            n_init=10,
        )
        return kmeans.fit_predict(reduced)

    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        n_components: int = 5,
        min_cluster_size: int = 15,
    ) -> np.ndarray:
        """Dimensionality reduction + density-based clustering.

        Primary path: UMAP → HDBSCAN.
        Fallback: PCA → KMeans.

        Returns array of integer cluster labels (-1 = noise).
        """
        if embeddings.shape[0] == 0:
            return np.array([], dtype=int)

        reduced = self._reduce_embeddings(embeddings, n_components=n_components)
        return self._cluster_reduced(reduced, min_cluster_size=min_cluster_size)

    @staticmethod
    def extract_topic_keywords(
        texts: Sequence[str],
        labels: np.ndarray,
        top_n: int = 10,
    ) -> Dict[int, List[Tuple[str, float]]]:
        """Extract top TF-IDF keywords per cluster.

        Returns ``{cluster_id: [(word, score), ...]}``.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        unique_labels = sorted(set(int(l) for l in labels) - {-1})
        if not unique_labels:
            return {}

        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        clean_texts = [str(t or "") for t in texts]
        tfidf = vectorizer.fit_transform(clean_texts)
        feature_names = np.array(vectorizer.get_feature_names_out())

        keywords: Dict[int, List[Tuple[str, float]]] = {}
        for lbl in unique_labels:
            mask = np.array(labels) == lbl
            if not mask.any():
                keywords[lbl] = []
                continue
            centroid = tfidf[mask].mean(axis=0)
            if hasattr(centroid, "A1"):
                centroid = centroid.A1
            else:
                centroid = np.asarray(centroid).ravel()
            top_idx = np.argsort(centroid)[::-1][:top_n]
            keywords[lbl] = [
                (str(feature_names[i]), float(centroid[i])) for i in top_idx if centroid[i] > 0
            ]

        return keywords

    def label_topics(self, topic_keywords: Dict[int, List[Tuple[str, float]]]) -> Dict[int, str]:
        """Create human-readable labels from the top 3 keywords of each topic."""
        labels: Dict[int, str] = {}
        for cluster_id, kws in topic_keywords.items():
            if not kws:
                labels[cluster_id] = f"topic_{cluster_id}"
            else:
                labels[cluster_id] = " / ".join(w for w, _ in kws[:3])
        return labels

    def assign_topics_to_comments(
        self,
        comments_df: pd.DataFrame,
        embeddings: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Full topic pipeline: embed → cluster → extract keywords → label → assign.

        Returns a DataFrame with *comment_id*, *topic_id*, *topic_label*.
        """
        self._check_columns(
            comments_df, ["comment_id", "comment_text"], "assign_topics_to_comments"
        )

        if embeddings is None:
            embeddings = self.generate_comment_embeddings(comments_df)

        texts = comments_df["comment_text"].tolist()
        labels = self.cluster_embeddings(embeddings)
        keywords = self.extract_topic_keywords(texts, labels)
        topic_labels = self.label_topics(keywords)

        result = comments_df[["comment_id"]].copy()
        result["topic_id"] = labels
        result["topic_label"] = result["topic_id"].map(topic_labels)
        result.loc[result["topic_id"] == -1, "topic_label"] = "noise"
        return result

    def get_topic_distribution(self, comments_df: pd.DataFrame) -> pd.DataFrame:
        """Return frequency distribution of topics."""
        self._check_columns(
            comments_df, ["topic_label"], "get_topic_distribution"
        )
        dist = (
            comments_df["topic_label"]
            .value_counts()
            .reset_index()
        )
        dist.columns = ["topic_label", "count"]
        dist["proportion"] = dist["count"] / dist["count"].sum()
        return dist

    # ------------------------------------------------------------------
    # Temporal topic drift
    # ------------------------------------------------------------------

    def compute_topic_timeseries(
        self,
        comments_df: pd.DataFrame,
        window: str = "W",
    ) -> pd.DataFrame:
        """Aggregate topic frequencies per time *window*.

        Requires *published_at* and *topic_label* columns.
        """
        self._check_columns(
            comments_df,
            ["published_at", "topic_label"],
            "compute_topic_timeseries",
        )

        df = comments_df.copy()
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
        df["time_bucket"] = df["published_at"].dt.to_period(window).astype(str)

        pivot = (
            df.groupby(["time_bucket", "topic_label"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )
        return pivot

    @staticmethod
    def identify_topic_trends(
        topic_timeseries: pd.DataFrame,
    ) -> Dict[str, str]:
        """Use linear-regression slopes to classify topic trends.

        Returns ``{topic_label: "emerging" | "declining" | "stable"}``.
        """
        from sklearn.linear_model import LinearRegression

        if topic_timeseries.empty:
            return {}

        x = np.arange(len(topic_timeseries)).reshape(-1, 1)
        trends: Dict[str, str] = {}

        for topic in topic_timeseries.columns:
            y = topic_timeseries[topic].values
            if y.sum() == 0:
                trends[str(topic)] = "stable"
                continue
            model = LinearRegression()
            model.fit(x, y)
            slope = model.coef_[0]
            scale = y.mean() if y.mean() != 0 else 1.0
            norm_slope = slope / scale

            if norm_slope > 0.05:
                trends[str(topic)] = "emerging"
            elif norm_slope < -0.05:
                trends[str(topic)] = "declining"
            else:
                trends[str(topic)] = "stable"

        return trends

    @staticmethod
    def compute_topic_drift(
        comments_df: pd.DataFrame,
        window1: str,
        window2: str,
    ) -> float:
        """Cosine distance between topic distributions in two time windows.

        Requires *published_at* and *topic_label* columns.
        *window1* / *window2* are ISO-format date strings or period strings.
        """
        if "published_at" not in comments_df.columns or "topic_label" not in comments_df.columns:
            raise KeyError("compute_topic_drift requires 'published_at' and 'topic_label' columns.")

        df = comments_df.copy()
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

        def _distribution(subset: pd.DataFrame) -> pd.Series:
            if subset.empty:
                return pd.Series(dtype=float)
            counts = subset["topic_label"].value_counts()
            return counts / counts.sum()

        try:
            mask1 = df["published_at"] < pd.Timestamp(window2)
            mask2 = (df["published_at"] >= pd.Timestamp(window2)) & (
                df["published_at"] < pd.Timestamp.now()
            )
            # If window1/window2 passed as date strings, use them directly
            t1 = pd.Timestamp(window1)
            t2 = pd.Timestamp(window2)
            mask1 = df["published_at"] < t1
            mask2 = (df["published_at"] >= t1) & (df["published_at"] < t2)
        except Exception:
            # Treat as period strings
            df["period"] = df["published_at"].dt.to_period("W").astype(str)
            mask1 = df["period"] == window1
            mask2 = df["period"] == window2

        dist1 = _distribution(df[mask1])
        dist2 = _distribution(df[mask2])

        all_topics = sorted(set(dist1.index) | set(dist2.index))
        v1 = np.array([dist1.get(t, 0.0) for t in all_topics])
        v2 = np.array([dist2.get(t, 0.0) for t in all_topics])

        if v1.sum() == 0 or v2.sum() == 0:
            return 1.0

        # Cosine similarity → distance
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0:
            return 1.0
        similarity = dot / norm
        return float(1.0 - similarity)

    # ------------------------------------------------------------------
    # Discourse-level sentiment report
    # ------------------------------------------------------------------

    def discourse_sentiment_report(
        self,
        comments_df: pd.DataFrame,
        videos_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Comprehensive sentiment report at video, channel, topic, and time-window levels.

        Returns
        -------
        dict
            Keys: *video_sentiment*, *channel_sentiment*, *time_sentiment*,
            *topic_sentiment*, *overall_sentiment*.
        """
        report: Dict[str, Any] = {}

        # Overall
        report["overall_sentiment"] = {
            "mean": float(comments_df["sentiment_score"].mean()),
            "std": float(comments_df["sentiment_score"].std()),
            "n_comments": len(comments_df),
        }

        # Per video
        try:
            report["video_sentiment"] = self.aggregate_sentiment_by_video(comments_df)
        except Exception:
            logger.exception("Could not compute video-level sentiment.")
            report["video_sentiment"] = None

        # Per channel
        try:
            report["channel_sentiment"] = self.aggregate_sentiment_by_channel(
                videos_df, comments_df
            )
        except Exception:
            logger.exception("Could not compute channel-level sentiment.")
            report["channel_sentiment"] = None

        # Per time window
        try:
            report["time_sentiment"] = self.aggregate_sentiment_by_time(comments_df)
        except Exception:
            logger.exception("Could not compute time-level sentiment.")
            report["time_sentiment"] = None

        # Per topic (if topic labels present)
        report["topic_sentiment"] = None
        if "topic_label" in comments_df.columns:
            try:
                report["topic_sentiment"] = self.aggregate_sentiment_by_topic(
                    comments_df,
                    comments_df[["comment_id", "topic_label"]].rename(
                        columns={comments_df.columns[0]: "comment_id"}
                    )
                    if "comment_id" not in comments_df.columns
                    else comments_df[["comment_id", "topic_label"]],
                )
            except Exception:
                logger.exception("Could not compute topic-level sentiment.")

        return report

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        videos_df: pd.DataFrame,
        comments_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Execute the complete NLP pipeline.

        Steps
        -----
        1. Sentiment analysis on comments
        2. Comment embeddings
        3. Video title & description embeddings
        4. Topic modeling
        5. Temporal topic drift
        6. Discourse-level sentiment report

        Returns a dict with all intermediate and final results.
        """
        results: Dict[str, Any] = {
            "sentiment": None,
            "comment_embeddings": None,
            "title_embeddings": None,
            "description_embeddings": None,
            "topic_assignments": None,
            "topic_timeseries": None,
            "topic_trends": None,
            "topic_drift": None,
            "discourse_report": None,
        }

        logger.info("=== Step 1: Sentiment analysis ===")
        try:
            sentiment_df = self.analyze_sentiment(comments_df["comment_text"].tolist())
            comments_df = comments_df.copy()
            for col in sentiment_df.columns:
                comments_df[col] = sentiment_df[col].values
            results["sentiment"] = sentiment_df
            logger.info("Sentiment analysis complete (%d comments).", len(comments_df))
        except Exception:
            logger.exception("Sentiment analysis failed.")

        logger.info("=== Step 2: Comment embeddings ===")
        try:
            results["comment_embeddings"] = self.generate_comment_embeddings(comments_df)
            logger.info("Comment embeddings generated.")
        except Exception:
            logger.exception("Comment embeddings failed.")

        logger.info("=== Step 3: Video title & description embeddings ===")
        try:
            results["title_embeddings"] = self.generate_title_embeddings(videos_df)
            results["description_embeddings"] = self.generate_description_embeddings(videos_df)
            logger.info("Video embeddings generated.")
        except Exception:
            logger.exception("Video embeddings failed.")

        logger.info("=== Step 4: Topic modeling ===")
        try:
            topic_df = self.assign_topics_to_comments(comments_df, results["comment_embeddings"])
            comments_df["topic_id"] = topic_df["topic_id"].values
            comments_df["topic_label"] = topic_df["topic_label"].values
            results["topic_assignments"] = topic_df
            results["topic_distribution"] = self.get_topic_distribution(comments_df)
            logger.info("Topic modeling complete (%d topics).", topic_df["topic_label"].nunique())
        except Exception:
            logger.exception("Topic modeling failed.")

        logger.info("=== Step 5: Temporal topic drift ===")
        try:
            results["topic_timeseries"] = self.compute_topic_timeseries(comments_df)
            if results["topic_timeseries"] is not None:
                results["topic_trends"] = self.identify_topic_trends(
                    results["topic_timeseries"]
                )
            logger.info("Temporal topic analysis complete.")
        except Exception:
            logger.exception("Temporal topic drift analysis failed.")

        logger.info("=== Step 6: Discourse-level sentiment ===")
        try:
            results["discourse_report"] = self.discourse_sentiment_report(
                comments_df, videos_df
            )
            logger.info("Discourse sentiment report generated.")
        except Exception:
            logger.exception("Discourse sentiment report failed.")

        logger.info("=== Full NLP pipeline complete ===")
        return results


# -----------------------------------------------------------------------
# __main__
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the NLP pipeline on processed data.")
    parser.add_argument(
        "--videos",
        default=os.path.join(CONFIG["PROCESSED_DATA_DIR"], "videos.parquet"),
        help="Path to videos parquet file.",
    )
    parser.add_argument(
        "--comments",
        default=os.path.join(CONFIG["PROCESSED_DATA_DIR"], "comments.parquet"),
        help="Path to comments parquet file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit comments to N rows (0 = no limit).",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(CONFIG["PROCESSED_DATA_DIR"], "nlp_results"),
        help="Directory to save results.",
    )
    args = parser.parse_args()

    # Load data
    try:
        videos_df = pd.read_parquet(args.videos)
        logger.info("Loaded %d videos from %s", len(videos_df), args.videos)
    except FileNotFoundError:
        logger.error("Videos file not found: %s", args.videos)
        raise SystemExit(1)
    except Exception as exc:
        logger.error("Failed to read videos file: %s", exc)
        raise SystemExit(1)

    try:
        comments_df = pd.read_parquet(args.comments)
        if args.limit > 0:
            comments_df = comments_df.head(args.limit)
        logger.info("Loaded %d comments from %s", len(comments_df), args.comments)
    except FileNotFoundError:
        logger.error("Comments file not found: %s", args.comments)
        raise SystemExit(1)
    except Exception as exc:
        logger.error("Failed to read comments file: %s", exc)
        raise SystemExit(1)

    # Basic validation
    required_comment_cols = {"comment_id", "comment_text", "published_at", "video_id"}
    missing_comment = required_comment_cols - set(comments_df.columns)
    if missing_comment:
        logger.warning(
            "Comments DataFrame missing expected columns: %s. "
            "Pipeline may produce incomplete results.",
            missing_comment,
        )

    required_video_cols = {"video_id", "title", "description", "channel_id"}
    missing_video = required_video_cols - set(videos_df.columns)
    if missing_video:
        logger.warning(
            "Videos DataFrame missing expected columns: %s. "
            "Pipeline may produce incomplete results.",
            missing_video,
        )

    # Run pipeline
    pipeline = NLPPipeline(batch_size=512)
    results = pipeline.run_full_pipeline(videos_df, comments_df)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    for key, value in results.items():
        if value is None:
            logger.info("Skipping %s (None)", key)
            continue
        try:
            if isinstance(value, pd.DataFrame):
                out_path = os.path.join(args.output_dir, f"{key}.parquet")
                value.to_parquet(out_path, index=False)
                logger.info("Saved %s → %s", key, out_path)
            elif isinstance(value, np.ndarray):
                out_path = os.path.join(args.output_dir, f"{key}.npy")
                np.save(out_path, value)
                logger.info("Saved %s → %s", key, out_path)
            elif isinstance(value, dict):
                out_path = os.path.join(args.output_dir, f"{key}.json")
                # Convert non-serializable items to strings
                import json

                serializable = {
                    k: v if isinstance(v, (str, int, float, bool, type(None), list, dict))
                    else str(v)
                    for k, v in value.items()
                }
                with open(out_path, "w") as f:
                    json.dump(serializable, f, indent=2, default=str)
                logger.info("Saved %s → %s", key, out_path)
            else:
                logger.info("Skipping %s (unsupported type %s)", key, type(value))
        except Exception:
            logger.exception("Failed to save %s", key)

    logger.info("NLP pipeline finished. Results saved to %s", args.output_dir)
