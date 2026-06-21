"""
Feature engineering module for digital media analytics.

Constructs a unified feature matrix from preprocessed video metadata,
comment data, NLP-derived sentiment/topic results, and embedding vectors.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import entropy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "RANDOM_STATE": 42,
    "PROCESSED_DATA_DIR": "./data/processed",
}

# ---------------------------------------------------------------------------
# FeatureEngineer
# ---------------------------------------------------------------------------


class FeatureEngineer:
    """Engineers features from preprocessed YouTube analytics data.

    Builds a unified feature matrix combining numerical metrics, temporal
    patterns, text-derived attributes, interaction terms, and reduced
    embedding vectors. Includes feature selection routines for
    dimensionality management.

    Parameters
    ----------
    random_state : int
        Seed for reproducible operations.
    """

    _VIEWS_CANDIDATES = ("views", "view_count", "Views", "View_Count", "VIEWS")
    _LIKES_CANDIDATES = ("likes", "like_count", "Likes", "Like_Count", "LIKES")
    _COMMENTS_CANDIDATES = ("comments", "comment_count", "Comments", "Comment_Count", "COMMENTS")

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self._feature_metadata: Dict[str, List[str]] = {}
        self._omissions: List[str] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_columns(self, df: pd.DataFrame, candidates: Tuple[str, ...]) -> str:
        """Return the first matching column name from *candidates*, falling
        back to the first candidate.  Logs a warning if no match is found."""
        for c in candidates:
            if c in df.columns:
                return c
        fallback = candidates[0]
        logger.warning(
            "Could not find any of %s in DataFrame columns %s; using '%s'.",
            candidates,
            list(df.columns),
            fallback,
        )
        return fallback

    @staticmethod
    def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        denom = denominator.replace(0, np.nan)
        return numerator / denom.where(pd.notna(denom), other=1.0)

    @staticmethod
    def _entropy(series: pd.Series) -> float:
        probs = series.value_counts(normalize=True)
        if len(probs) <= 1:
            return 0.0
        return float(entropy(probs.values, base=2))

    # ==================================================================
    # 1. NUMERICAL FEATURES
    # ==================================================================

    def create_numerical_features(
        self,
        videos_df: pd.DataFrame,
        comments_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compute per-video numerical features.

        Parameters
        ----------
        videos_df : pd.DataFrame
            Preprocessed videos table. Must contain ``video_id``.
        comments_df : pd.DataFrame or None
            Optional comments table used to derive a per-video comment count
            when ``comment_count`` is not present in *videos_df*.

        Returns
        -------
        pd.DataFrame
            Indexed by ``video_id`` with columns:
            ``views``, ``likes``, ``comments``, ``engagement_rate``,
            ``like_rate``, ``comment_rate``, ``video_age_days``,
            ``views_per_day``, ``comments_per_day``, ``log_views``,
            ``log_likes``, ``log_comments``.
        """
        df = videos_df.copy()

        if "video_id" not in df.columns:
            raise KeyError("videos_df must contain 'video_id' column.")

        view_col = self._resolve_columns(df, self._VIEWS_CANDIDATES)
        like_col = self._resolve_columns(df, self._LIKES_CANDIDATES)
        comm_col = self._resolve_columns(df, self._COMMENTS_CANDIDATES)

        features = pd.DataFrame(index=df.index)
        features["video_id"] = df["video_id"]

        # --- raw counts ---
        for src_col, out_col in [(view_col, "views"), (like_col, "likes"), (comm_col, "comments")]:
            if src_col in df.columns:
                features[out_col] = pd.to_numeric(df[src_col], errors="coerce").fillna(0)
            else:
                features[out_col] = 0.0
                self._omissions.append(f"numerical_{out_col}")

        # --- engagement / rate metrics (verify from preprocessed data) ---
        for col in ["engagement_rate", "like_rate", "comment_rate"]:
            if col in df.columns:
                features[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            else:
                features[col] = 0.0
                self._omissions.append(f"numerical_{col}")

        # --- video_age_days ---
        if "video_age_days" in df.columns:
            features["video_age_days"] = pd.to_numeric(
                df["video_age_days"], errors="coerce"
            ).fillna(0).clip(lower=0)
        else:
            features["video_age_days"] = 0.0
            self._omissions.append("numerical_video_age_days")

        # --- derived features ---
        age_safe = features["video_age_days"].clip(lower=1)
        features["views_per_day"] = self._safe_div(features["views"], age_safe)
        features["comments_per_day"] = self._safe_div(features["comments"], age_safe)

        features["log_views"] = np.log1p(features["views"])
        features["log_likes"] = np.log1p(features["likes"])
        features["log_comments"] = np.log1p(features["comments"])

        features = features.set_index("video_id")
        features.index.name = "video_id"
        return features

    # ==================================================================
    # 2. TEMPORAL FEATURES
    # ==================================================================

    def create_temporal_features(self, videos_df: pd.DataFrame) -> pd.DataFrame:
        """Extract and compute temporal features from *videos_df*.

        Parameters
        ----------
        videos_df : pd.DataFrame
            Must contain ``video_id``, ``channel_id``, ``published_at``.
            ``views`` and ``video_age_days`` are required for
            *growth_rate*.

        Returns
        -------
        pd.DataFrame
            Indexed by ``video_id`` with columns: ``hour``,
            ``day_of_week``, ``month``, ``year``, ``is_weekend``,
            ``time_since_channel_start``, ``growth_rate``,
            ``posting_frequency``.
        """
        df = videos_df.copy()

        if "video_id" not in df.columns:
            raise KeyError("videos_df must contain 'video_id' column.")

        features = pd.DataFrame(index=df.index)
        features["video_id"] = df["video_id"]

        # --- ensure basic temporal columns ---
        ts_col = None
        for c in ["published_at", "publishedAt", "publish_date", "published_date"]:
            if c in df.columns:
                ts_col = c
                break

        if ts_col is not None:
            ts = pd.to_datetime(df[ts_col], errors="coerce")
            features["hour"] = ts.dt.hour.fillna(0).astype(int)
            features["day_of_week"] = ts.dt.dayofweek.fillna(0).astype(int)
            features["month"] = ts.dt.month.fillna(1).astype(int)
            features["year"] = ts.dt.year.fillna(2024).astype(int)
            features["is_weekend"] = (features["day_of_week"].isin([5, 6])).astype(int)
        else:
            # attempt to read pre-computed temporal columns
            for col, default in [
                ("hour", 0), ("day_of_week", 0), ("month", 1),
                ("year", 2024), ("is_weekend", 0),
            ]:
                if col in df.columns:
                    features[col] = pd.to_numeric(df[col], errors="coerce").fillna(default).astype(int)
                else:
                    features[col] = default
                    self._omissions.append(f"temporal_{col}")

        # --- time_since_channel_start ---
        channel_col = self._resolve_columns(df, ("channel_id", "channelId", "Channel_ID"))
        if ts_col is not None and channel_col in df.columns:
            channel_dates = df.groupby(channel_col)[ts_col].transform("min")
            channel_dates = pd.to_datetime(channel_dates, errors="coerce")
            features["time_since_channel_start"] = (
                (pd.to_datetime(df[ts_col], errors="coerce") - channel_dates).dt.days
            ).fillna(0).clip(lower=0)
        else:
            features["time_since_channel_start"] = 0
            self._omissions.append("temporal_time_since_channel_start")

        # --- growth_rate ---
        view_col = self._resolve_columns(df, self._VIEWS_CANDIDATES)
        if view_col in df.columns:
            views = pd.to_numeric(df[view_col], errors="coerce").fillna(0)
        else:
            views = pd.Series(0.0, index=df.index)

        if "video_age_days" in df.columns:
            age = pd.to_numeric(df["video_age_days"], errors="coerce").fillna(1).clip(lower=1)
        else:
            age = pd.Series(1.0, index=df.index)

        video_growth = self._safe_div(views, age)

        if channel_col in df.columns:
            channel_mean_growth = video_growth.groupby(df[channel_col]).transform("mean")
            features["growth_rate"] = self._safe_div(video_growth, channel_mean_growth.clip(lower=1e-6)).fillna(1.0)
        else:
            features["growth_rate"] = 1.0

        # --- posting_frequency (count of videos per channel in trailing 7 days) ---
        if ts_col is not None and channel_col in df.columns:
            ts_sorted = df.sort_values([channel_col, ts_col]).copy()
            ts_vals = pd.to_datetime(ts_sorted[ts_col], errors="coerce").values

            def _count_last_7d(grp_idx: np.ndarray) -> pd.Series:
                grp_ts = ts_vals[grp_idx]
                counts = np.ones(len(grp_ts), dtype=float)
                seven_days = np.timedelta64(7, "D")
                for i in range(len(grp_ts)):
                    counts[i] = np.sum(
                        (grp_ts >= grp_ts[i] - seven_days) & (grp_ts <= grp_ts[i])
                    )
                return pd.Series(counts, index=grp_idx)

            grouped_idx = ts_sorted.groupby(channel_col, sort=False).indices
            freq_map = pd.concat(
                [_count_last_7d(idx_arr) for idx_arr in grouped_idx.values()]
            )
            features["posting_frequency"] = freq_map.reindex(df.index).fillna(1).astype(float)
        else:
            features["posting_frequency"] = 1.0
            self._omissions.append("temporal_posting_frequency")

        features = features.set_index("video_id")
        features.index.name = "video_id"
        return features

    # ==================================================================
    # 3. TEXT FEATURES
    # ==================================================================

    def create_text_features(
        self,
        videos_df: pd.DataFrame,
        comments_df: Optional[pd.DataFrame] = None,
        nlp_results: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Compute per-video features derived from text metadata and NLP results.

        Parameters
        ----------
        videos_df : pd.DataFrame
            Must contain ``video_id``, ``title``, ``description``.
        comments_df : pd.DataFrame or None
            Comments table (used for comment count aggregation and
            comment-length stats).
        nlp_results : dict or None
            Expected keys: ``sentiment`` (per-comment DataFrame with
            ``video_id``, ``sentiment_score``), and ``topic_assignments``
            (per-comment DataFrame with ``video_id``, ``topic_id``).

        Returns
        -------
        pd.DataFrame
            Indexed by ``video_id``.
        """
        df = videos_df.copy()
        if "video_id" not in df.columns:
            raise KeyError("videos_df must contain 'video_id' column.")

        features = pd.DataFrame(index=df.index)
        features["video_id"] = df["video_id"]

        # --- title & description lengths ---
        for src, out in [("title", "title_length"), ("description", "description_length")]:
            if src in df.columns:
                features[out] = df[src].fillna("").astype(str).str.len()
            else:
                features[out] = 0
                self._omissions.append(f"text_{out}")

        # --- comment count ---
        comm_col = self._resolve_columns(df, self._COMMENTS_CANDIDATES)
        if comm_col in df.columns:
            features["comment_count"] = pd.to_numeric(df[comm_col], errors="coerce").fillna(0).astype(int)
        elif comments_df is not None and "video_id" in comments_df.columns:
            features["comment_count"] = (
                features["video_id"]
                .map(comments_df.groupby("video_id").size())
                .fillna(0)
                .astype(int)
            )
        else:
            features["comment_count"] = 0
            self._omissions.append("text_comment_count")

        # --- sentiment features (per video) ---
        sentiment_none = True
        if nlp_results is not None and "sentiment" in nlp_results:
            sent_df = nlp_results["sentiment"]
            if isinstance(sent_df, pd.DataFrame) and "video_id" in sent_df.columns:
                sentiment_none = False
                sent_col = (
                    "sentiment_score"
                    if "sentiment_score" in sent_df.columns
                    else sent_df.columns[1]
                )
                grp = sent_df.groupby("video_id")[sent_col]
                features["avg_comment_sentiment"] = features["video_id"].map(grp.mean()).fillna(0.0)
                features["sentiment_variance"] = features["video_id"].map(grp.var()).fillna(0.0)

        if sentiment_none:
            features["avg_comment_sentiment"] = 0.0
            features["sentiment_variance"] = 0.0
            self._omissions.append("text_sentiment")

        # --- topic features ---
        topic_none = True
        if nlp_results is not None and "topic_assignments" in nlp_results:
            topic_df = nlp_results["topic_assignments"]
            if isinstance(topic_df, pd.DataFrame) and "video_id" in topic_df.columns:
                topic_none = False
                tid = "topic_id" if "topic_id" in topic_df.columns else "topic_label"
                if tid in topic_df.columns:
                    # most common topic per video
                    mode_map = (
                        topic_df.groupby("video_id")[tid]
                        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else -1)
                    )
                    features["topic_id"] = features["video_id"].map(mode_map).fillna(-1).astype(int)

                    # entropy of topic distribution per video
                    def _topic_entropy(x: pd.Series) -> float:
                        return self._entropy(x)

                    ent_map = topic_df.groupby("video_id")[tid].agg(_topic_entropy)
                    features["topic_entropy"] = features["video_id"].map(ent_map).fillna(0.0)

        if topic_none:
            features["topic_id"] = -1
            features["topic_entropy"] = 0.0
            self._omissions.append("text_topics")

        # --- avg_comment_length ---
        if comments_df is not None and "video_id" in comments_df.columns:
            text_col = None
            for c in ["text_cleaned", "comment_text", "text_display", "text"]:
                if c in comments_df.columns:
                    text_col = c
                    break
            if text_col:
                comments_df = comments_df.copy()
                comments_df["_clen"] = comments_df[text_col].fillna("").astype(str).str.len()
                len_map = comments_df.groupby("video_id")["_clen"].mean()
                features["avg_comment_length"] = features["video_id"].map(len_map).fillna(0.0)
            else:
                features["avg_comment_length"] = 0.0
                self._omissions.append("text_avg_comment_length")
        else:
            features["avg_comment_length"] = 0.0
            self._omissions.append("text_avg_comment_length")

        features = features.set_index("video_id")
        features.index.name = "video_id"
        return features

    # ==================================================================
    # 4. INTERACTION AND NORMALIZED FEATURES
    # ==================================================================

    def create_interaction_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction and normalized performance features.

        Parameters
        ----------
        features_df : pd.DataFrame
            Feature matrix (indexed by ``video_id``) that includes
            ``channel_id``, ``engagement_rate``, ``views``, ``likes``,
            ``comments``, ``avg_comment_sentiment``, and optionally
            ``topic_id``.

        Returns
        -------
        pd.DataFrame
            Indexed by ``video_id`` with added columns:
            ``normalized_engagement_index``,
            ``channel_relative_performance``,
            ``topic_relative_performance``,
            ``sentiment_engagement_interaction``,
            ``views_likes_ratio``, ``comments_to_likes_ratio``.
        """
        df = features_df.copy()

        # --- normalized_engagement_index (z-score per channel) ---
        if "channel_id" in df.columns and "engagement_rate" in df.columns:
            grp = df.groupby("channel_id")["engagement_rate"]
            mean_eng = grp.transform("mean")
            std_eng = grp.transform("std").clip(lower=1e-8)
            df["normalized_engagement_index"] = ((df["engagement_rate"] - mean_eng) / std_eng).fillna(0)
        else:
            df["normalized_engagement_index"] = 0.0

        # --- channel_relative_performance ---
        if "channel_id" in df.columns and "views" in df.columns:
            ch_mean = df.groupby("channel_id")["views"].transform("mean")
            df["channel_relative_performance"] = self._safe_div(df["views"], ch_mean.clip(lower=1e-6)).fillna(1.0)
        else:
            df["channel_relative_performance"] = 1.0

        # --- topic_relative_performance ---
        if "topic_id" in df.columns and "engagement_rate" in df.columns:
            topic_mean = df.groupby("topic_id")["engagement_rate"].transform("mean")
            df["topic_relative_performance"] = self._safe_div(
                df["engagement_rate"], topic_mean.clip(lower=1e-6)
            ).fillna(1.0)
        else:
            df["topic_relative_performance"] = 1.0

        # --- sentiment_engagement_interaction ---
        if "avg_comment_sentiment" in df.columns and "engagement_rate" in df.columns:
            df["sentiment_engagement_interaction"] = (
                df["avg_comment_sentiment"] * df["engagement_rate"]
            )
        else:
            df["sentiment_engagement_interaction"] = 0.0

        # --- views_likes_ratio ---
        if "likes" in df.columns and "views" in df.columns:
            df["views_likes_ratio"] = self._safe_div(df["likes"], df["views"].clip(lower=1))
        else:
            df["views_likes_ratio"] = 0.0

        # --- comments_to_likes_ratio ---
        if "comments" in df.columns and "likes" in df.columns:
            df["comments_to_likes_ratio"] = self._safe_div(df["comments"], df["likes"].clip(lower=1))
        else:
            df["comments_to_likes_ratio"] = 0.0

        return df

    # ==================================================================
    # 5. DIMENSIONALITY REDUCTION
    # ==================================================================

    def reduce_embeddings(
        self,
        embedding_matrix: np.ndarray,
        n_components: int = 10,
        method: str = "pca",
    ) -> pd.DataFrame:
        """Reduce high-dimensional embeddings via PCA or UMAP.

        Parameters
        ----------
        embedding_matrix : np.ndarray
            Shape ``(n_samples, n_features)``.
        n_components : int
            Target dimensionality.
        method : str, optional
            ``"pca"`` or ``"umap"``.

        Returns
        -------
        pd.DataFrame
            Columns ``emb_0``, ``emb_1``, ..., ``emb_{n_components-1}``.
        """
        n_samples = embedding_matrix.shape[0]
        n_components = min(n_components, n_samples, embedding_matrix.shape[1])
        if n_components < 1:
            n_components = 1

        if method == "umap":
            try:
                import umap

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    reducer = umap.UMAP(
                        n_components=n_components,
                        random_state=self.random_state,
                        n_neighbors=min(15, n_samples - 1),
                        min_dist=0.1,
                    )
                reduced = reducer.fit_transform(embedding_matrix)
            except ImportError:
                logger.warning("umap-learn not available; falling back to PCA.")
                method = "pca"
            except Exception as exc:
                logger.warning("UMAP reduction failed (%s); falling back to PCA.", exc)
                method = "pca"

        if method == "pca":
            from sklearn.decomposition import PCA

            n_components = min(n_components, min(embedding_matrix.shape) - 1, embedding_matrix.shape[0])
            if n_components < 1:
                n_components = 1
            pca = PCA(n_components=n_components, random_state=self.random_state)
            reduced = pca.fit_transform(embedding_matrix)

        cols = [f"emb_{i}" for i in range(reduced.shape[1])]
        return pd.DataFrame(reduced, columns=cols)

    def add_embedding_features(
        self,
        features_df: pd.DataFrame,
        title_embeddings: Optional[np.ndarray] = None,
        desc_embeddings: Optional[np.ndarray] = None,
        comment_embeddings: Optional[np.ndarray] = None,
        n_components: int = 10,
    ) -> pd.DataFrame:
        """Reduce each embedding type and join columns to *features_df*.

        Missing embedding arrays are silently skipped.

        Parameters
        ----------
        features_df : pd.DataFrame
            Feature matrix indexed by ``video_id``.
        title_embeddings : np.ndarray or None
        desc_embeddings : np.ndarray or None
        comment_embeddings : np.ndarray or None
        n_components : int
            Dimensionality after reduction.

        Returns
        -------
        pd.DataFrame
            Augmented *features_df*.
        """
        df = features_df.copy()

        embedding_sources = {
            "title": title_embeddings,
            "desc": desc_embeddings,
            "comment": comment_embeddings,
        }

        for prefix, emb in embedding_sources.items():
            if emb is None or (isinstance(emb, np.ndarray) and emb.size == 0):
                continue
            if not isinstance(emb, np.ndarray) or emb.ndim != 2:
                continue
            try:
                n_comp = min(n_components, emb.shape[0], emb.shape[1])
                reduced = self.reduce_embeddings(emb, n_components=n_comp, method="pca")
                # align index — assume same order as features_df
                if len(reduced) != len(df):
                    logger.warning(
                        "Embedding row count %d != features row count %d for '%s'; skipping.",
                        len(reduced),
                        len(df),
                        prefix,
                    )
                    continue
                reduced.columns = [f"{prefix}_{c}" for c in reduced.columns]
                reduced.index = df.index
                df = pd.concat([df, reduced], axis=1)
            except Exception as exc:
                logger.warning("Failed to add '%s' embeddings: %s", prefix, exc)
                self._omissions.append(f"embeddings_{prefix}")

        return df

    # ==================================================================
    # 6. FEATURE SELECTION
    # ==================================================================

    def select_features_by_correlation(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        threshold: float = 0.9,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Remove highly correlated features (Pearson > *threshold*).

        When a pair exceeds the threshold the feature with the lower
        absolute correlation to *target* is dropped.

        Parameters
        ----------
        features_df : pd.DataFrame
            Numeric-only feature matrix.
        target : pd.Series
            Target vector aligned with *features_df*.
        threshold : float
            Correlation cutoff.

        Returns
        -------
        tuple
            ``(reduced_features_df, list_of_removed_columns)``.
        """
        numeric = features_df.select_dtypes(include=[np.number])
        if numeric.empty:
            return features_df, []

        corr_matrix = numeric.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop: List[str] = []
        target_corr = numeric.apply(lambda col: col.corr(target) if col.std() > 0 else 0.0)

        for column in upper.columns:
            if column in to_drop:
                continue
            high_corr = upper.index[upper[column] > threshold].tolist()
            for other in high_corr:
                if other in to_drop or other == column:
                    continue
                drop_col = column if abs(target_corr[other]) >= abs(target_corr[column]) else other
                if drop_col not in to_drop:
                    to_drop.append(drop_col)

        if to_drop:
            logger.info(
                "Dropping %d highly-correlated features (threshold=%.2f): %s",
                len(to_drop),
                threshold,
                to_drop[:10],
            )
            return features_df.drop(columns=to_drop, errors="ignore"), to_drop

        return features_df, []

    def select_features_by_importance(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        n_features: int = 20,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Select top *n_features* via Random Forest importance.

        Parameters
        ----------
        features_df : pd.DataFrame
            Numeric feature matrix.
        target : pd.Series
            Target vector.
        n_features : int
            Number of features to retain.

        Returns
        -------
        tuple
            ``(reduced_features_df, list_of_selected_feature_names)``.
        """
        from sklearn.ensemble import RandomForestRegressor

        numeric = features_df.select_dtypes(include=[np.number])
        if numeric.empty or numeric.shape[1] <= n_features:
            return features_df, list(numeric.columns)

        # fill remaining NaNs
        numeric = numeric.fillna(numeric.median())

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(numeric, target.fillna(target.median()))

        importances = pd.Series(model.feature_importances_, index=numeric.columns)
        selected = importances.nlargest(n_features).index.tolist()

        logger.info("Selected top %d features by importance.", len(selected))
        return features_df[selected], selected

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Return features grouped by category.

        Returns
        -------
        dict
            Keys: ``numerical``, ``temporal``, ``text``, ``interaction``,
            ``embedding``. Values are lists of column names.
        """
        return self._feature_metadata.copy()

    # ==================================================================
    # 7. BUILD FEATURE MATRIX
    # ==================================================================

    def build_feature_matrix(
        self,
        videos_df: pd.DataFrame,
        comments_df: Optional[pd.DataFrame] = None,
        nlp_results: Optional[Dict[str, Any]] = None,
        embeddings_dict: Optional[Dict[str, Optional[np.ndarray]]] = None,
        target_col: str = "views",
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """Orchestrate all feature engineering steps into a unified matrix.

        Parameters
        ----------
        videos_df : pd.DataFrame
            Preprocessed videos table.
        comments_df : pd.DataFrame or None
            Preprocessed comments table.
        nlp_results : dict or None
            NLP pipeline output (sentiment, topic_assignments, etc.).
        embeddings_dict : dict or None
            Expected keys: ``title_embeddings``, ``desc_embeddings``,
            ``comment_embeddings`` (each an np.ndarray or None).
        target_col : str
            Column name to use as the target variable (default ``"views"``).

        Returns
        -------
        tuple
            ``(feature_matrix, feature_metadata)`` where *feature_matrix*
            is a DataFrame indexed by ``video_id`` with all engineered
            features plus the target column, and *feature_metadata* is a
            dict mapping category names to lists of feature column names.
        """
        self._omissions.clear()
        self._feature_metadata.clear()

        logger.info("Building feature matrix ...")

        # --- 1. Numerical ---
        num_features = self.create_numerical_features(videos_df, comments_df)
        self._feature_metadata["numerical"] = list(num_features.columns)
        logger.info("  Numerical features: %d", len(num_features.columns))

        # --- 2. Temporal ---
        temp_features = self.create_temporal_features(videos_df)
        self._feature_metadata["temporal"] = list(temp_features.columns)
        logger.info("  Temporal features: %d", len(temp_features.columns))

        # --- 3. Text ---
        text_features = self.create_text_features(videos_df, comments_df, nlp_results)
        self._feature_metadata["text"] = list(text_features.columns)
        logger.info("  Text features: %d", len(text_features.columns))

        # --- Merge base feature sets ---
        feature_matrix = (
            num_features
            .join(temp_features, how="left")
            .join(text_features, how="left")
        )

        # Attach channel_id for interaction features
        channel_col = self._resolve_columns(videos_df, ("channel_id", "channelId", "Channel_ID"))
        if channel_col in videos_df.columns:
            feature_matrix["channel_id"] = (
                videos_df.set_index("video_id").reindex(feature_matrix.index)[channel_col]
            )

        # --- 4. Interaction ---
        feature_matrix = self.create_interaction_features(feature_matrix)
        inter_cols = [c for c in feature_matrix.columns if c not in (
            set(num_features.columns)
            | set(temp_features.columns)
            | set(text_features.columns)
            | {"channel_id"}
        )]
        self._feature_metadata["interaction"] = inter_cols
        logger.info("  Interaction features: %d", len(inter_cols))

        # --- 5. Embeddings ---
        emb_cols: List[str] = []
        if embeddings_dict:
            title_emb = embeddings_dict.get("title_embeddings")
            desc_emb = embeddings_dict.get("desc_embeddings", embeddings_dict.get("description_embeddings"))
            comm_emb = embeddings_dict.get("comment_embeddings")

            if title_emb is not None or desc_emb is not None or comm_emb is not None:
                n_before = feature_matrix.shape[1]
                feature_matrix = self.add_embedding_features(
                    feature_matrix,
                    title_embeddings=title_emb,
                    desc_embeddings=desc_emb,
                    comment_embeddings=comm_emb,
                )
                emb_cols = [c for c in feature_matrix.columns[n_before:]]
                self._feature_metadata["embedding"] = emb_cols
                logger.info("  Embedding features: %d", len(emb_cols))
        else:
            self._omissions.append("embeddings_all")
            self._feature_metadata["embedding"] = []

        # --- 6. Attach target ---
        view_col = self._resolve_columns(videos_df, self._VIEWS_CANDIDATES)
        if target_col in videos_df.columns:
            target_series = pd.to_numeric(videos_df[target_col], errors="coerce").fillna(0)
        elif view_col in videos_df.columns and target_col == "views":
            target_series = pd.to_numeric(videos_df[view_col], errors="coerce").fillna(0)
        else:
            # default: use log_views as proxy
            target_series = feature_matrix.get("log_views", pd.Series(0.0, index=feature_matrix.index))

        if "video_id" in videos_df.columns:
            target_series.index = videos_df["video_id"].values
        target_series = target_series.reindex(feature_matrix.index)
        feature_matrix["target"] = target_series.values

        # --- Drop channel_id from final matrix ---
        feature_matrix = feature_matrix.drop(columns=["channel_id"], errors="ignore")

        logger.info(
            "Feature matrix built: %d rows × %d columns (target included).",
            feature_matrix.shape[0],
            feature_matrix.shape[1],
        )

        if self._omissions:
            logger.warning("Omissions / missing data: %s", self._omissions)

        return feature_matrix, self._feature_metadata

    # ==================================================================
    # 8. PERSISTENCE
    # ==================================================================

    def save_features(
        self,
        feature_matrix: pd.DataFrame,
        filename: str = "feature_matrix.parquet",
    ) -> str:
        """Save the feature matrix to a Parquet file.

        Parameters
        ----------
        feature_matrix : pd.DataFrame
            Feature matrix to persist.
        filename : str
            Output file name.

        Returns
        -------
        str
            Full path to the saved file.
        """
        output_dir = CONFIG["PROCESSED_DATA_DIR"]
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        feature_matrix.to_parquet(path, index=True)
        logger.info("Feature matrix saved to %s", path)
        return path


# ======================================================================
# __main__
# ======================================================================

if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    data_dir = CONFIG["PROCESSED_DATA_DIR"]

    # ---- load processed data ----
    videos_path = os.path.join(data_dir, "videos_processed.parquet")
    comments_path = os.path.join(data_dir, "comments_processed.parquet")

    videos_df: Optional[pd.DataFrame] = None
    comments_df: Optional[pd.DataFrame] = None

    try:
        videos_df = pd.read_parquet(videos_path)
        logger.info("Loaded videos: %s (%s)", videos_path, videos_df.shape)
    except FileNotFoundError:
        logger.warning("Videos file not found: %s. Trying raw ...", videos_path)
        alt = os.path.join(data_dir, "videos.parquet")
        try:
            videos_df = pd.read_parquet(alt)
            logger.info("Loaded videos (alt): %s", alt)
        except FileNotFoundError:
            pass

    try:
        comments_df = pd.read_parquet(comments_path)
        logger.info("Loaded comments: %s (%s)", comments_path, comments_df.shape)
    except FileNotFoundError:
        logger.warning("Comments file not found: %s. Trying raw ...", comments_path)
        alt = os.path.join(data_dir, "comments.parquet")
        try:
            comments_df = pd.read_parquet(alt)
            logger.info("Loaded comments (alt): %s", alt)
        except FileNotFoundError:
            pass

    if videos_df is None or videos_df.empty:
        logger.warning("No video data available. Generating minimal example ...")
        rng = np.random.default_rng(CONFIG["RANDOM_STATE"])
        n = 60
        videos_df = pd.DataFrame({
            "video_id": [f"ex_{i:04d}" for i in range(n)],
            "channel_id": rng.choice(["ch_a", "ch_b", "ch_c"], size=n),
            "title": [f"Example Video {i}" for i in range(n)],
            "description": [f"Description for video {i}" for i in range(n)],
            "published_at": pd.to_datetime(
                rng.integers(
                    pd.Timestamp("2024-01-01").value // 10**9,
                    pd.Timestamp("2025-06-01").value // 10**9,
                    size=n,
                ),
                unit="s",
            ),
            "views": rng.lognormal(mean=9, sigma=1.5, size=n).astype(int),
            "likes": rng.lognormal(mean=6, sigma=1.8, size=n).astype(int),
            "comments": rng.lognormal(mean=4, sigma=1.5, size=n).astype(int),
            "engagement_rate": rng.beta(2, 6, size=n),
            "video_age_days": rng.integers(1, 365, size=n),
        })

    # ---- load NLP results if available ----
    nlp_results: Optional[Dict[str, Any]] = None
    nlp_dir = os.path.join(data_dir, "nlp_results")
    if os.path.isdir(nlp_dir):
        nlp_results = {}
        for key in ["sentiment", "topic_assignments"]:
            parquet_path = os.path.join(nlp_dir, f"{key}.parquet")
            if os.path.isfile(parquet_path):
                try:
                    nlp_results[key] = pd.read_parquet(parquet_path)
                    logger.info("Loaded NLP result '%s' from %s", key, parquet_path)
                except Exception as exc:
                    logger.warning("Failed to load NLP result '%s': %s", key, exc)
        if not nlp_results:
            nlp_results = None

    # ---- load embeddings if available ----
    embeddings_dict: Optional[Dict[str, Optional[np.ndarray]]] = None
    for emb_key in ["title_embeddings", "description_embeddings", "comment_embeddings"]:
        npy_path = os.path.join(nlp_dir, f"{emb_key}.npy") if os.path.isdir(nlp_dir) else ""
        if npy_path and os.path.isfile(npy_path):
            if embeddings_dict is None:
                embeddings_dict = {}
            try:
                embeddings_dict[emb_key] = np.load(npy_path)
                logger.info("Loaded %s from %s (shape %s)", emb_key, npy_path, embeddings_dict[emb_key].shape)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", emb_key, exc)

    # ---- build feature matrix ----
    engineer = FeatureEngineer(random_state=CONFIG["RANDOM_STATE"])
    feature_matrix, metadata = engineer.build_feature_matrix(
        videos_df=videos_df,
        comments_df=comments_df,
        nlp_results=nlp_results,
        embeddings_dict=embeddings_dict,
    )

    # ---- save ----
    engineer.save_features(feature_matrix)

    meta_path = os.path.join(data_dir, "feature_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({k: v for k, v in metadata.items()}, f, indent=2, default=str)
    logger.info("Feature metadata saved to %s", meta_path)

    omissions_path = os.path.join(data_dir, "feature_omissions.json")
    with open(omissions_path, "w") as f:
        json.dump(engineer._omissions, f, indent=2)
    logger.info("Omissions log saved to %s", omissions_path)

    logger.info("Feature engineering complete.")
