"""
Advanced data preprocessing module for digital media analytics.

Provides comprehensive numerical, temporal, categorical, and text preprocessing
capabilities, plus schema validation for YouTube channel, video, comment, and
reply datasets.
"""

import re
import sys
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "PROCESSED_DATA_DIR": "./data/processed",
    "RANDOM_STATE": 42,
    "BATCH_SIZE": 512,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# NLTK stopwords — loaded lazily to avoid import-time overhead
_STOPWORDS: Optional[set] = None


def _get_stopwords() -> set:
    """Return the set of English NLTK stopwords, downloading them if needed."""
    global _STOPWORDS
    if _STOPWORDS is None:
        try:
            import nltk  # noqa: F811

            try:
                _STOPWORDS = set(nltk.corpus.stopwords.words("english"))
            except LookupError:
                nltk.download("stopwords", quiet=True)
                _STOPWORDS = set(nltk.corpus.stopwords.words("english"))
        except Exception as exc:
            logger.warning("Could not load NLTK stopwords: %s. Using empty set.", exc)
            _STOPWORDS = set()
    return _STOPWORDS


_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"  # misc
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols etc
    "\U0001FA70-\U0001FAFF"  # symbols extended-A
    "\U00002600-\U000026FF"  # misc symbols
    "\U00002B50"             # star
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"             # zero-width joiner
    "]+",
    flags=re.UNICODE,
)

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

_SPECIAL_PUNCTUATION = re.compile(r"[^a-zA-Z0-9\s\.\,\!\?\;\:\-\(\)\[\]]")


# ---------------------------------------------------------------------------
# AdvancedPreprocessing
# ---------------------------------------------------------------------------


class AdvancedPreprocessing:
    """Comprehensive preprocessing pipeline for digital media analytics.

    Handles numerical feature engineering, temporal parsing, categorical
    encoding, text cleaning/tokenization/lemmatization, outlier detection,
    and referential-integrity schema validation.

    Parameters
    ----------
    random_state : int
        Seed for reproducible random operations (default 42).
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.onehot_encoders: Dict[str, OneHotEncoder] = {}
        self.scaler = StandardScaler()
        self._spacy_nlp: Optional[Any] = None

    # ------------------------------------------------------------------
    # spaCy model lazy loader
    # ------------------------------------------------------------------

    @property
    def _nlp(self) -> Any:
        """Lazy-load the spaCy English model with graceful fallback."""
        if self._spacy_nlp is not None:
            return self._spacy_nlp
        try:
            import spacy

            self._spacy_nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model 'en_core_web_sm' successfully.")
        except OSError:
            logger.warning(
                "spaCy model 'en_core_web_sm' not found. Install it with:\n"
                "    python -m spacy download en_core_web_sm\n"
                "Falling back to simple whitespace tokenization."
            )
            self._spacy_nlp = None
        except Exception as exc:
            logger.warning(
                "Could not load spaCy model: %s. Falling back to simple "
                "whitespace tokenization.",
                exc,
            )
            self._spacy_nlp = None
        return self._spacy_nlp

    # ==================================================================
    # 1. NUMERICAL PREPROCESSING
    # ==================================================================

    def preprocess_numerical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert count columns to numeric, impute missing values, create
        engagement metrics, and append missingness indicator flags.

        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame containing at minimum ``views``, ``likes``,
            ``comments`` columns (case-insensitive matched).

        Returns
        -------
        pd.DataFrame
            DataFrame with additional engineered columns:
            ``{col}_missing`` indicators, ``engagement_rate``,
            ``like_rate``, ``comment_rate``, ``virality_score``.
        """
        df = df.copy()

        # ---- locate key columns (case-insensitive) ----
        col_map = {}
        for target in ["views", "likes", "comments"]:
            for col in df.columns:
                if col.lower() == target:
                    col_map[target] = col
                    break

        # ---- convert to numeric ----
        for target, col in col_map.items():
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # ---- missingness flags + median imputation for ALL numeric cols ----
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                flag_name = f"{col}_missing"
                df[flag_name] = df[col].isna().astype(int)
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)

        # ---- engagement metrics (only if all 3 key cols present) ----
        if all(k in col_map for k in ["views", "likes", "comments"]):
            v_col = col_map["views"]
            l_col = col_map["likes"]
            c_col = col_map["comments"]

            df["engagement_rate"] = np.where(
                df[v_col] > 0,
                (df[l_col] + df[c_col]) / df[v_col],
                0.0,
            )
            df["like_rate"] = np.where(df[v_col] > 0, df[l_col] / df[v_col], 0.0)
            df["comment_rate"] = np.where(df[v_col] > 0, df[c_col] / df[v_col], 0.0)

            # ---- virality score (min-max normalised) ----
            views_norm = self._minmax_norm(df[v_col])
            likes_norm = self._minmax_norm(df[l_col])
            comments_norm = self._minmax_norm(df[c_col])
            engagement_norm = self._minmax_norm(df["engagement_rate"])

            df["virality_score"] = (
                0.3 * views_norm
                + 0.25 * likes_norm
                + 0.25 * comments_norm
                + 0.2 * engagement_norm
            )

        return df

    @staticmethod
    def _minmax_norm(series: pd.Series) -> pd.Series:
        mmax = series.max()
        mmin = series.min()
        if mmax == mmin:
            return pd.Series(0.5, index=series.index)
        return (series - mmin) / (mmax - mmin)

    def detect_outliers_zscore(
        self, df: pd.DataFrame, columns: List[str], threshold: float = 3.0
    ) -> pd.Series:
        """Z-score based outlier detection.

        Parameters
        ----------
        df : pd.DataFrame
        columns : list of str
            Numeric columns to examine.
        threshold : float
            Observations with |z-score| > threshold are flagged (default 3).

        Returns
        -------
        pd.Series
            Boolean mask where ``True`` indicates an outlier in *any* of the
            specified columns.
        """
        from scipy import stats

        mask = pd.Series(False, index=df.index)
        for col in columns:
            col_data = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())
            if col_data.std() == 0:
                continue
            z_scores = np.abs(stats.zscore(col_data, nan_policy="omit"))
            mask |= pd.Series(z_scores > threshold, index=df.index)
        return mask

    def detect_outliers_iqr(
        self, df: pd.DataFrame, columns: List[str], multiplier: float = 1.5
    ) -> pd.Series:
        """IQR-based outlier detection.

        Parameters
        ----------
        df : pd.DataFrame
        columns : list of str
            Numeric columns to examine.
        multiplier : float
            IQR multiplier (default 1.5).

        Returns
        -------
        pd.Series
            Boolean mask where ``True`` indicates an outlier in *any* column.
        """
        mask = pd.Series(False, index=df.index)
        for col in columns:
            col_data = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr
            mask |= (col_data < lower) | (col_data > upper)
        return mask

    def detect_outliers_isolation_forest(
        self,
        df: pd.DataFrame,
        columns: List[str],
        contamination: float = 0.05,
    ) -> pd.Series:
        """Isolation Forest outlier detection.

        Parameters
        ----------
        df : pd.DataFrame
        columns : list of str
            Numeric columns to examine.
        contamination : float
            Expected proportion of outliers (default 0.05).

        Returns
        -------
        pd.Series
            Boolean mask where ``True`` indicates an outlier.
        """
        sub = df[columns].select_dtypes(include=[np.number]).fillna(0)
        if sub.empty:
            return pd.Series(False, index=df.index)
        model = IsolationForest(
            contamination=contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        preds = model.fit_predict(sub)
        return pd.Series(preds == -1, index=df.index)

    def label_outliers(
        self, df: pd.DataFrame, columns: List[str]
    ) -> pd.DataFrame:
        """Combine z-score, IQR, and Isolation Forest to label outliers.

        A row is labelled outlier (1) if at least two of the three methods
        flag it.

        Parameters
        ----------
        df : pd.DataFrame
        columns : list of str
            Numeric columns.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with an added ``outlier_label`` column (0 or 1).
        """
        df = df.copy()
        z = self.detect_outliers_zscore(df, columns)
        i = self.detect_outliers_iqr(df, columns)
        f = self.detect_outliers_isolation_forest(df, columns)
        df["outlier_label"] = ((z.astype(int) + i.astype(int) + f.astype(int)) >= 2).astype(int)
        return df

    # ==================================================================
    # 2. TEMPORAL PREPROCESSING
    # ==================================================================

    def preprocess_temporal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse ``published_at``, extract temporal features, compute video age.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a ``published_at`` column (or recognised variants:
            ``publishedAt``, ``publish_date``, ``published_date``).

        Returns
        -------
        pd.DataFrame
            DataFrame augmented with ``hour``, ``day_of_week``, ``month``,
            ``year``, ``is_weekend``, ``video_age_days``.
        """
        df = df.copy()

        # ---- locate datetime column ----
        ts_col = None
        candidates = ["published_at", "publishedAt", "publish_date", "published_date"]
        for c in candidates:
            if c in df.columns:
                ts_col = c
                break
        if ts_col is None:
            logger.warning("No recognised timestamp column found; skipping temporal preprocessing.")
            return df

        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df["hour"] = df[ts_col].dt.hour
        df["day_of_week"] = df[ts_col].dt.dayofweek
        df["month"] = df[ts_col].dt.month
        df["year"] = df[ts_col].dt.year
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # video age relative to now, or max date in column if historical data
        ref_date = pd.Timestamp.now(tz=df[ts_col].dt.tz) if pd.api.types.is_datetime64tz_dtype(df[ts_col]) else pd.Timestamp.now()
        max_date = df[ts_col].max()
        if pd.notna(max_date):
            ref_date = max_date
        df["video_age_days"] = (ref_date - df[ts_col]).dt.days

        return df

    def create_temporal_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set ``published_at`` as the DatetimeIndex.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a ``published_at`` column.

        Returns
        -------
        pd.DataFrame
            Indexed DataFrame sorted by the new index.
        """
        df = df.copy()
        ts_col = None
        for c in ["published_at", "publishedAt", "publish_date", "published_date"]:
            if c in df.columns:
                ts_col = c
                break
        if ts_col is None:
            raise KeyError("No timestamp column found to set as index.")
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col])
        df = df.set_index(ts_col).sort_index()
        return df

    def aggregate_by_window(
        self, df: pd.DataFrame, window: str = "D"
    ) -> pd.DataFrame:
        """Aggregate numeric columns by a time window.

        Parameters
        ----------
        df : pd.DataFrame
            Must be temporally indexed (via ``create_temporal_index``).
        window : str
            Pandas offset alias: ``'D'``, ``'W'``, ``'M'``, ``'Q'``, ``'Y'``.

        Returns
        -------
        pd.DataFrame
            Resampled DataFrame with sums of ``views``, ``likes``, ``comments``
            (matched case-insensitively).
        """
        agg_map = {}
        for target in ["views", "likes", "comments"]:
            for col in df.columns:
                if col.lower() == target:
                    agg_map[col] = "sum"
                    break
        if not agg_map:
            logger.warning("No numeric count columns found for aggregation.")
            return df.resample(window).sum()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in agg_map:
                agg_map[col] = "mean"
        return df.resample(window).agg(agg_map)

    # ==================================================================
    # 3. CATEGORICAL PREPROCESSING
    # ==================================================================

    def preprocess_categorical(
        self, df: pd.DataFrame, columns: List[str], max_categories: int = 20
    ) -> pd.DataFrame:
        """Label-encode low-cardinality columns and one-hot-encode the rest.

        Parameters
        ----------
        df : pd.DataFrame
        columns : list of str
            Categorical columns to encode.
        max_categories : int
            Columns with ≤ this many unique values are label-encoded;
            the rest are one-hot encoded (default 20).

        Returns
        -------
        pd.DataFrame
            DataFrame with categorical columns encoded. Original columns are
            dropped after encoding.
        """
        df = df.copy()
        for col in columns:
            if col not in df.columns:
                continue
            df[col] = df[col].astype(str).replace({"nan": None, "None": None})
            df[col] = df[col].fillna("MISSING")
            n_unique = df[col].nunique()
            if n_unique <= max_categories:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoded = ohe.fit_transform(df[[col]])
                ohe_cols = [f"{col}_{cat}" for cat in ohe.categories_[0]]
                encoded_df = pd.DataFrame(
                    encoded, columns=ohe_cols, index=df.index
                )
                df = df.drop(columns=[col]).join(encoded_df)
                self.onehot_encoders[col] = ohe
        return df

    # ==================================================================
    # 4. TEXT PREPROCESSING
    # ==================================================================

    @staticmethod
    def clean_text(text: Any) -> str:
        """Clean a single text string.

        Steps applied in order:
        1. Coerce to string, handle None/NaN.
        2. Lowercase.
        3. Remove URLs.
        4. Remove emojis (unicode-based regex).
        5. Strip special characters while keeping basic punctuation.

        Parameters
        ----------
        text : any
            Input text (str, None, NaN, numeric, …).

        Returns
        -------
        str
            Cleaned text.
        """
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return ""
        text = str(text)
        text = text.lower()
        text = _URL_PATTERN.sub(" ", text)
        text = _EMOJI_PATTERN.sub(" ", text)
        text = _SPECIAL_PUNCTUATION.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def remove_stopwords(tokens: List[str]) -> List[str]:
        """Filter out English stopwords from a token list.

        Parameters
        ----------
        tokens : list of str
            Pre-tokenized words.

        Returns
        -------
        list of str
            Tokens with stopwords removed.
        """
        sw = _get_stopwords()
        return [t for t in tokens if t not in sw]

    def preprocess_text_series(
        self,
        series: pd.Series,
        batch_size: Optional[int] = None,
    ) -> pd.Series:
        """Apply ``clean_text`` to every element of a pandas Series.

        Uses ``tqdm`` for a progress bar.  Handles None/NaN gracefully.

        Parameters
        ----------
        series : pd.Series
            Text data.
        batch_size : int or None
            If provided, process in chunks of this size (unused in current
            implementation beyond tqdm configuration — kept for API
            compatibility).

        Returns
        -------
        pd.Series
            Cleaned strings.
        """
        total = len(series)
        desc = "Cleaning text"
        results = []
        for idx, val in tqdm(
            series.items(), total=total, desc=desc, file=sys.stderr
        ):
            results.append(self.clean_text(val))
        return pd.Series(results, index=series.index, name=series.name)

    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize and lemmatize text using spaCy; falls back to ``str.split``.

        Parameters
        ----------
        text : str
            Raw input string.

        Returns
        -------
        list of str
            Lemmatized tokens (or lowercased whitespace-split tokens on
            fallback).
        """
        nlp = self._nlp
        if nlp is not None:
            try:
                doc = nlp(text)
                return [
                    token.lemma_.lower()
                    for token in doc
                    if not token.is_space and not token.is_punct and token.lemma_ != ""
                ]
            except Exception as exc:
                logger.debug("spaCy tokenization failed: %s. Using split fallback.", exc)
        # Fallback
        tokens = text.split()
        return [t.strip().lower() for t in tokens if t.strip()]

    # ==================================================================
    # 5. SCHEMA VALIDATION
    # ==================================================================

    _CHANNELS_REQUIRED = [
        ("channel_id", "object"),
        ("channel_title", "object"),
        ("subscriber_count", "number"),
        ("video_count", "number"),
    ]

    _VIDEOS_REQUIRED = [
        ("video_id", "object"),
        ("channel_id", "object"),
        ("title", "object"),
        ("published_at", "datetime"),
        ("views", "number"),
    ]

    _COMMENTS_REQUIRED = [
        ("comment_id", "object"),
        ("video_id", "object"),
        ("text_display", "object"),
    ]

    def validate_channels_schema(self, channels_df: pd.DataFrame) -> Dict[str, Any]:
        """Check required columns and types on the channels DataFrame.

        Returns
        -------
        dict
            Keys: ``valid`` (bool), ``missing_columns``, ``type_mismatches``,
            ``row_count``, ``message``.
        """
        return self._validate_df_schema(channels_df, self._CHANNELS_REQUIRED, "channels")

    def validate_videos_schema(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """Check required columns and types on the videos DataFrame."""
        return self._validate_df_schema(videos_df, self._VIDEOS_REQUIRED, "videos")

    def validate_comments_schema(self, comments_df: pd.DataFrame) -> Dict[str, Any]:
        """Check required columns and types on the comments DataFrame."""
        return self._validate_df_schema(comments_df, self._COMMENTS_REQUIRED, "comments")

    def _validate_df_schema(
        self, df: pd.DataFrame, required: List[Tuple[str, str]], name: str
    ) -> Dict[str, Any]:
        missing = [col for col, _ in required if col not in df.columns]
        type_mismatches = []
        type_map = {
            "object": "object",
            "number": "number",
            "datetime": "datetime64",
        }
        for col, dtype in required:
            if col not in df.columns:
                continue
            if dtype == "number":
                if not pd.api.types.is_numeric_dtype(df[col]):
                    type_mismatches.append((col, dtype, str(df[col].dtype)))
            elif dtype == "datetime":
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    type_mismatches.append((col, dtype, str(df[col].dtype)))

        valid = len(missing) == 0 and len(type_mismatches) == 0
        return {
            "table": name,
            "valid": valid,
            "row_count": len(df),
            "missing_columns": missing,
            "type_mismatches": type_mismatches,
            "message": "OK" if valid else f"Schema issues: missing={missing}, mismatches={type_mismatches}",
        }

    def check_missing_foreign_keys(
        self,
        videos_df: pd.DataFrame,
        channels_df: pd.DataFrame,
        comments_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Verify referential integrity between videos ↔ channels and
        comments ↔ videos.

        Returns
        -------
        dict
            ``valid``, ``orphan_video_channel``, ``orphan_comment_video``,
            ``message``.
        """
        result: Dict[str, Any] = {"valid": True, "orphan_video_channel": [], "orphan_comment_video": []}

        if "channel_id" in videos_df.columns and "channel_id" in channels_df.columns:
            valid_channels = set(channels_df["channel_id"].dropna().unique())
            video_channels = set(videos_df["channel_id"].dropna().unique())
            orphans = video_channels - valid_channels
            result["orphan_video_channel"] = sorted(orphans)

        if "video_id" in comments_df.columns and "video_id" in videos_df.columns:
            valid_videos = set(videos_df["video_id"].dropna().unique())
            comment_videos = set(comments_df["video_id"].dropna().unique())
            orphans = comment_videos - valid_videos
            result["orphan_comment_video"] = sorted(orphans)

        result["valid"] = len(result["orphan_video_channel"]) == 0 and len(result["orphan_comment_video"]) == 0
        result["message"] = (
            "OK"
            if result["valid"]
            else f"Orphans: video→channel={result['orphan_video_channel']}, comment→video={result['orphan_comment_video']}"
        )
        return result

    @staticmethod
    def check_duplicate_primary_keys(
        df: pd.DataFrame, key_column: str
    ) -> Dict[str, Any]:
        """Check for duplicate values in a primary key column.

        Returns
        -------
        dict
            ``valid`` (bool), ``duplicate_count``, ``duplicate_keys`` (list),
            ``message``.
        """
        if key_column not in df.columns:
            return {
                "valid": False,
                "duplicate_count": 0,
                "duplicate_keys": [],
                "message": f"Column '{key_column}' not found.",
            }
        dup_mask = df[key_column].duplicated(keep=False)
        dup_vals = df.loc[dup_mask, key_column].unique().tolist()
        return {
            "valid": len(dup_vals) == 0,
            "duplicate_count": len(dup_vals),
            "duplicate_keys": sorted(dup_vals),
            "message": "OK" if len(dup_vals) == 0 else f"Duplicate keys: {dup_vals}",
        }

    @staticmethod
    def check_null_critical_fields(
        df: pd.DataFrame, critical_columns: List[str]
    ) -> Dict[str, Any]:
        """Check for nulls in user-specified critical fields.

        Returns
        -------
        dict
            ``valid``, ``null_counts`` (dict column → count), ``message``.
        """
        null_counts = {}
        for col in critical_columns:
            if col in df.columns:
                null_counts[col] = int(df[col].isna().sum())
            else:
                null_counts[col] = -1  # column not present
        total_nulls = sum(v for v in null_counts.values() if v > 0)
        return {
            "valid": total_nulls == 0,
            "null_counts": null_counts,
            "message": "OK" if total_nulls == 0 else f"Nulls found in critical fields: {null_counts}",
        }

    def validate_schema(
        self,
        channels_df: pd.DataFrame,
        videos_df: pd.DataFrame,
        comments_df: pd.DataFrame,
        replies_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Run all validation checks and return a consolidated report.

        Parameters
        ----------
        channels_df : pd.DataFrame
        videos_df : pd.DataFrame
        comments_df : pd.DataFrame
        replies_df : pd.DataFrame
            Replies table (checked for common-sense columns but no strict
            schema enforced).

        Returns
        -------
        dict
            ``overall_valid``, ``channels``, ``videos``, ``comments``,
            ``replies``, ``foreign_keys``, ``duplicates``, ``critical_nulls``.
        """
        results: Dict[str, Any] = {}

        results["channels"] = self.validate_channels_schema(channels_df)
        results["videos"] = self.validate_videos_schema(videos_df)
        results["comments"] = self.validate_comments_schema(comments_df)

        results["replies"] = {
            "valid": True,
            "row_count": len(replies_df),
            "message": "OK (no strict schema for replies)",
        }

        results["foreign_keys"] = self.check_missing_foreign_keys(
            videos_df, channels_df, comments_df
        )

        dup_results = {}
        for name, df, pk in [
            ("channels", channels_df, "channel_id"),
            ("videos", videos_df, "video_id"),
            ("comments", comments_df, "comment_id"),
        ]:
            dup_results[name] = self.check_duplicate_primary_keys(df, pk)

        if "reply_id" in replies_df.columns:
            dup_results["replies"] = self.check_duplicate_primary_keys(replies_df, "reply_id")
        else:
            dup_results["replies"] = {"valid": True, "duplicate_count": 0, "duplicate_keys": [], "message": "OK"}

        results["duplicates"] = dup_results

        results["critical_nulls"] = {
            "channels": self.check_null_critical_fields(channels_df, ["channel_id", "channel_title"]),
            "videos": self.check_null_critical_fields(videos_df, ["video_id", "channel_id", "title"]),
            "comments": self.check_null_critical_fields(comments_df, ["comment_id", "video_id", "text_display"]),
        }

        results["overall_valid"] = all(
            results[k]["valid"]
            for k in ["channels", "videos", "comments", "foreign_keys"]
        )
        return results

    # ==================================================================
    # 6. FULL PIPELINE
    # ==================================================================

    def run_full_pipeline(
        self,
        channels_df: pd.DataFrame,
        videos_df: pd.DataFrame,
        comments_df: pd.DataFrame,
        replies_df: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """Execute all preprocessing steps end-to-end.

        Steps
        -----
        1. Schema validation.
        2. Numerical preprocessing on videos.
        3. Outlier labelling on videos.
        4. Temporal preprocessing on videos.
        5. Categorical preprocessing on channels & videos.
        6. Text preprocessing on comments & replies.

        Parameters
        ----------
        channels_df : pd.DataFrame
        videos_df : pd.DataFrame
        comments_df : pd.DataFrame
        replies_df : pd.DataFrame

        Returns
        -------
        dict
            ``{'channels': ..., 'videos': ..., 'comments': ..., 'replies': ...,
            'validation_report': ...}``
        """
        print("=" * 60)
        print("Starting full preprocessing pipeline\n")

        # 1. Validate
        print("[1/5] Running schema validation...")
        validation = self.validate_schema(channels_df, videos_df, comments_df, replies_df)
        print(f"  Overall valid: {validation['overall_valid']}")
        if not validation["overall_valid"]:
            logger.warning("Schema validation issues detected. See report for details.")

        # 2. Process videos (numerical + temporal + outliers)
        print("[2/5] Preprocessing videos (numerical → outliers → temporal)...")
        videos = self.preprocess_numerical(videos_df)
        num_cols = videos.select_dtypes(include=[np.number]).columns.tolist()
        # exclude rate columns and missing flags from outlier detection
        outlier_cols = [
            c
            for c in num_cols
            if not c.endswith("_missing")
            and c
            not in {
                "engagement_rate",
                "like_rate",
                "comment_rate",
                "virality_score",
                "outlier_label",
                "hour",
                "day_of_week",
                "month",
                "year",
                "is_weekend",
                "video_age_days",
            }
        ]
        videos = self.label_outliers(videos, outlier_cols)
        videos = self.preprocess_temporal(videos)
        print(f"  Videos: {videos.shape}")

        # 3. Process channels (categorical only)
        print("[3/5] Preprocessing channels (categorical)...")
        cat_cols_channels = channels_df.select_dtypes(include=["object", "category"]).columns.tolist()
        channels = self.preprocess_categorical(channels_df, cat_cols_channels)
        print(f"  Channels: {channels.shape}")

        # 4. Process comments (text)
        print("[4/5] Preprocessing comments (text)...")
        comments = comments_df.copy()
        if "text_display" in comments.columns:
            comments["text_cleaned"] = self.preprocess_text_series(comments["text_display"])
            comments["text_tokens"] = comments["text_cleaned"].apply(self.tokenize_and_lemmatize)
            comments["text_tokens_nosw"] = comments["text_tokens"].apply(self.remove_stopwords)
        print(f"  Comments: {comments.shape}")

        # 5. Process replies (text)
        print("[5/5] Preprocessing replies (text)...")
        replies = replies_df.copy()
        if "text_display" in replies.columns:
            replies["text_cleaned"] = self.preprocess_text_series(replies["text_display"])
            replies["text_tokens"] = replies["text_cleaned"].apply(self.tokenize_and_lemmatize)
            replies["text_tokens_nosw"] = replies["text_tokens"].apply(self.remove_stopwords)
        print(f"  Replies: {replies.shape}")

        print("\nPipeline complete.")
        print("=" * 60)

        return {
            "channels": channels,
            "videos": videos,
            "comments": comments,
            "replies": replies,
            "validation_report": validation,
        }


# ======================================================================
# __main__
# ======================================================================

if __name__ == "__main__":
    import os

    data_dir = CONFIG["PROCESSED_DATA_DIR"]
    files = {
        "channels": os.path.join(data_dir, "channels.parquet"),
        "videos": os.path.join(data_dir, "videos.parquet"),
        "comments": os.path.join(data_dir, "comments.parquet"),
        "replies": os.path.join(data_dir, "replies.parquet"),
    }

    dfs: Dict[str, pd.DataFrame] = {}
    for name, path in files.items():
        try:
            dfs[name] = pd.read_parquet(path)
            print(f"Loaded {name}: {dfs[name].shape} from {path}")
        except FileNotFoundError:
            print(f"File not found: {path} — skipping {name}.")
            dfs[name] = pd.DataFrame()
        except Exception as exc:
            print(f"Error loading {path}: {exc} — skipping {name}.")
            dfs[name] = pd.DataFrame()

    # Run pipeline
    preprocessor = AdvancedPreprocessing(random_state=CONFIG["RANDOM_STATE"])
    result = preprocessor.run_full_pipeline(
        dfs["channels"], dfs["videos"], dfs["comments"], dfs["replies"]
    )

    # Save outputs
    output_dir = data_dir
    os.makedirs(output_dir, exist_ok=True)

    output_map = {
        "channels_processed": result["channels"],
        "videos_processed": result["videos"],
        "comments_processed": result["comments"],
        "replies_processed": result["replies"],
    }

    for name, out_df in output_map.items():
        out_path = os.path.join(output_dir, f"{name}.parquet")
        out_df.to_parquet(out_path, index=True)
        print(f"Saved {name} ({out_df.shape}) → {out_path}")

    # Save validation report as JSON
    import json

    report_path = os.path.join(output_dir, "validation_report.json")
    # Convert non-serializable objects to strings
    report_clean = {}
    for k, v in result["validation_report"].items():
        if isinstance(v, dict):
            report_clean[k] = {
                kk: (str(vv) if not isinstance(vv, (str, int, float, bool, list, dict, type(None))) else vv)
                for kk, vv in v.items()
            }
        else:
            report_clean[k] = str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v
    with open(report_path, "w") as f:
        json.dump(report_clean, f, indent=2, default=str)
    print(f"Saved validation report → {report_path}")
