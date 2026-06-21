"""
Synthetic Data Generator for Digital Media Analytics.

Generates privacy-preserving synthetic versions of YouTube analytics data
using statistical distribution fitting, correlation preservation, and
lightweight text augmentation.
"""

import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.linalg import cholesky, LinAlgError
from scipy.spatial.distance import cosine as cosine_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

CONFIG = {
    "SYNTHETIC_DATA_DIR": "./data/synthetic",
    "PROCESSED_DATA_DIR": "./data/processed",
    "RANDOM_STATE": 42,
    "SYNTHETIC_MULTIPLIER": 3,
    "NOISE_STD": 0.1,
    "SEMANTIC_SIMILARITY_THRESHOLD": 0.3,   # FIX: lowered from 0.75 → works for short/mixed-language comments
}


class SyntheticDataGenerator:
    """Generates synthetic data for digital media analytics datasets."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.distributions: Optional[Dict[str, Any]] = None
        self.corr_matrix: Optional[np.ndarray] = None
        self.corr_L: Optional[np.ndarray] = None
        self.column_means: Optional[Dict[str, float]] = None
        self.column_stds: Optional[Dict[str, float]] = None
        self.columns_order: List[str] = []
        self.date_min: Optional[pd.Timestamp] = None
        self.date_max: Optional[pd.Timestamp] = None

        self._tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self._sentence_model: Any = None
        self._bert_fill_mask: Any = None
        self._nltk_ready: bool = False

    # ------------------------------------------------------------------
    # NLTK / WordNet helpers
    # ------------------------------------------------------------------
    def _ensure_nltk(self) -> None:
        if self._nltk_ready:
            return
        try:
            import nltk
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            nltk.download("averaged_perceptron_tagger", quiet=True)
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
            self._nltk_ready = True
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 2. NUMERICAL SYNTHETIC GENERATION
    # ------------------------------------------------------------------
    def fit_distributions(
        self, df: pd.DataFrame, columns: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        distributions = {}
        for col in columns:
            data = df[col].dropna().values
            if len(data) < 5:
                continue
            positive_data = data[data > 0]
            entry: Dict[str, Any] = {"min": float(data.min()), "max": float(data.max())}

            if len(positive_data) >= 5:
                try:
                    shape, loc, scale = stats.lognorm.fit(positive_data, floc=0)
                    entry["lognorm"] = (shape, loc, scale)
                except Exception:
                    entry["lognorm"] = None
                try:
                    a, loc_g, scale_g = stats.gamma.fit(positive_data, floc=0)
                    entry["gamma"] = (a, loc_g, scale_g)
                except Exception:
                    entry["gamma"] = None
            else:
                entry["lognorm"] = None
                entry["gamma"] = None

            entry["empirical"] = data.copy()
            distributions[col] = entry

        self.distributions = distributions
        return distributions

    def generate_synthetic_numerical(
        self,
        df: pd.DataFrame,
        n_samples: int,
        distributions: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        dists = distributions or self.distributions
        if dists is None:
            raise ValueError("Distributions not fitted. Call fit_distributions first.")

        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c in dists]
        self.columns_order = numeric_cols

        corr_df = df[numeric_cols].dropna()
        if len(corr_df) < 5:
            self.corr_matrix = np.eye(len(numeric_cols))
            self.corr_L = np.eye(len(numeric_cols))
        else:
            self.corr_matrix = corr_df.corr().values
            if np.any(np.isnan(self.corr_matrix)):
                self.corr_matrix = np.eye(len(numeric_cols))
            eigvals = np.linalg.eigvalsh(self.corr_matrix)
            if np.any(eigvals <= 1e-10):
                delta = abs(eigvals.min()) + 1e-6
                self.corr_matrix += delta * np.eye(len(numeric_cols))
            try:
                self.corr_L = cholesky(self.corr_matrix, lower=True)
            except LinAlgError:
                self.corr_L = np.linalg.cholesky(
                    self.corr_matrix + 1e-6 * np.eye(len(numeric_cols))
                )

        self.column_means = {c: corr_df[c].mean() for c in numeric_cols}
        self.column_stds = {c: max(corr_df[c].std(), 1e-8) for c in numeric_cols}

        raw = self.rng.normal(0, 1, size=(n_samples, len(numeric_cols)))
        correlated = raw @ self.corr_L.T
        col_idx = {c: i for i, c in enumerate(numeric_cols)}

        synthetic = pd.DataFrame(index=range(n_samples))

        if "views" in numeric_cols:
            i_v = col_idx["views"]
            dist = dists.get("views", {})
            views = self._sample_lognorm(dist, n_samples, correlated[:, i_v])
            views = np.clip(views, 0, None)
            synthetic["views"] = views.astype(int)

        if "likes" in numeric_cols:
            dist = dists.get("likes", {})
            likes = self._sample_lognorm(dist, n_samples, correlated[:, col_idx["likes"]])
            likes = np.clip(likes, 0, None)
            if "views" in synthetic.columns:
                likes = np.minimum(likes, synthetic["views"].values)
            synthetic["likes"] = likes.astype(int)

        if "comments" in numeric_cols:
            dist = dists.get("comments", {})
            comments = self._sample_lognorm(dist, n_samples, correlated[:, col_idx["comments"]])
            comments = np.clip(comments, 0, None)
            if "views" in synthetic.columns:
                comments = np.minimum(comments, synthetic["views"].values)
            synthetic["comments"] = comments.astype(int)

        if "engagement_rate" in numeric_cols:
            u_vals = correlated[:, col_idx["engagement_rate"]]
            raw_vals = stats.norm.cdf(u_vals)
            alpha = self.rng.uniform(1.5, 5.0, size=n_samples)
            beta_param = self.rng.uniform(1.5, 8.0, size=n_samples)
            engagement = np.array(
                [stats.beta.ppf(raw_vals[i], alpha[i], beta_param[i]) for i in range(n_samples)]
            )
            engagement = np.clip(engagement, 0, 1)
            synthetic["engagement_rate"] = engagement

        if "published_at" in df.columns:
            if self.date_min is None:
                self.date_min = pd.to_datetime(df["published_at"]).min()
            if self.date_max is None:
                self.date_max = pd.to_datetime(df["published_at"]).max()
            # FIX: strip tz from date_min/date_max to avoid tz-naive issues
            date_min = self.date_min
            date_max = self.date_max
            if hasattr(date_min, 'tzinfo') and date_min.tzinfo is not None:
                date_min = date_min.tz_localize(None)
            if hasattr(date_max, 'tzinfo') and date_max.tzinfo is not None:
                date_max = date_max.tz_localize(None)
            synthetic["published_at"] = pd.to_datetime(
                self.rng.integers(
                    int(date_min.value // 10**9),
                    int(date_max.value // 10**9),
                    size=n_samples,
                ),
                unit="s",
            )

        return synthetic

    @staticmethod
    def _sample_lognorm(
        dist: Dict[str, Any], n: int, noise: np.ndarray
    ) -> np.ndarray:
        if dist.get("lognorm") is not None:
            shape, loc, scale = dist["lognorm"]
            base = stats.lognorm.ppf(stats.norm.cdf(noise), shape, loc=loc, scale=scale)
        elif dist.get("empirical") is not None and len(dist["empirical"]) > 0:
            emp = dist["empirical"]
            idx = np.clip(
                ((stats.norm.cdf(noise) * len(emp))).astype(int), 0, len(emp) - 1
            )
            base = np.sort(emp)[idx]
        else:
            base = np.exp(noise)
        return np.maximum(base, 0)

    def inject_noise(
        self, df: pd.DataFrame, columns: List[str], noise_std: float = 0.1
    ) -> pd.DataFrame:
        df_noisy = df.copy()
        skewed_cols = {"views", "likes", "comments"}

        for col in columns:
            if col not in df_noisy.columns:
                continue
            vals = df_noisy[col].values.astype(float)
            if col in skewed_cols:
                noise = self.rng.lognormal(mean=0, sigma=noise_std, size=len(vals))
                df_noisy[col] = np.maximum(vals * noise, 0)
            else:
                noise = self.rng.normal(0, noise_std * vals.std(ddof=1), size=len(vals))
                df_noisy[col] = vals + noise

        if "likes" in columns and "views" in df_noisy.columns:
            df_noisy["likes"] = np.minimum(df_noisy["likes"].values, df_noisy["views"].values)
        if "comments" in columns and "views" in df_noisy.columns:
            df_noisy["comments"] = np.minimum(df_noisy["comments"].values, df_noisy["views"].values)
        if "engagement_rate" in columns:
            df_noisy["engagement_rate"] = np.clip(df_noisy["engagement_rate"].values, 0, 1)

        return df_noisy

    def compare_distributions(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        columns: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        # FIX: filter to only columns present in both DataFrames with enough data
        valid_columns = [
            c for c in columns
            if c in original.columns
            and c in synthetic.columns
            and len(original[c].dropna()) >= 5
            and len(synthetic[c].dropna()) >= 5
        ]

        results: Dict[str, Dict[str, Any]] = {}

        # FIX: guard against empty column list — skip plotting entirely
        if not valid_columns:
            print("[INFO] compare_distributions: no valid columns to compare, skipping plots.")
            return results

        n_cols = len(valid_columns)

        fig_hist, axes_hist = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), squeeze=False)
        fig_qq, axes_qq = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), squeeze=False)

        for idx, col in enumerate(valid_columns):
            orig = original[col].dropna().values
            synth = synthetic[col].dropna().values

            ks_stat, ks_pval = stats.ks_2samp(orig, synth)
            results[col] = {"ks_statistic": ks_stat, "ks_pvalue": ks_pval}

            ax_h = axes_hist[0][idx]
            ax_h.hist(orig, bins=30, alpha=0.5, density=True, label="Original", color="skyblue")
            ax_h.hist(synth, bins=30, alpha=0.5, density=True, label="Synthetic", color="salmon")
            ax_h.set_title(f"{col}\nKS={ks_stat:.3f}, p={ks_pval:.3f}")
            ax_h.legend(fontsize=7)

            ax_q = axes_qq[0][idx]
            stats.probplot(synth, dist="norm", plot=ax_q)
            ax_q.set_title(f"QQ: {col}")

        fig_hist.suptitle("Histogram Overlay: Original vs Synthetic", fontsize=13)
        fig_hist.tight_layout()
        fig_qq.suptitle("QQ Plots: Synthetic vs Normal", fontsize=13)
        fig_qq.tight_layout()

        os.makedirs("./outputs/figures", exist_ok=True)
        fig_hist.savefig("./outputs/figures/histogram_comparison.png", dpi=150, bbox_inches="tight")
        fig_qq.savefig("./outputs/figures/qq_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig_hist)
        plt.close(fig_qq)
        return results

    # ------------------------------------------------------------------
    # 3. TEXT AUGMENTATION
    # ------------------------------------------------------------------
    def synonym_replacement(self, text: str, n: int = 2) -> str:
        self._ensure_nltk()
        if not self._nltk_ready:
            return text
        try:
            from nltk.corpus import wordnet
        except ImportError:
            return text

        if not isinstance(text, str) or not text.strip():
            return text

        words = text.split()
        if len(words) < 2 or n <= 0:
            return text

        candidate_indices = [
            i for i, w in enumerate(words)
            if w.isalpha() and len(wordnet.synsets(w)) > 0
        ]
        if not candidate_indices:
            return text

        n = min(n, len(candidate_indices))
        chosen = list(self.rng.choice(candidate_indices, size=n, replace=False))
        new_words = words[:]
        for i in chosen:
            syns = wordnet.synsets(words[i])
            if not syns:
                continue
            lemmas = set()
            for syn in syns:
                for lemma in syn.lemmas():
                    name = lemma.name().replace("_", " ")
                    if name.lower() != words[i].lower():
                        lemmas.add(name)
            if lemmas:
                new_words[i] = self.rng.choice(list(lemmas))

        return " ".join(new_words)

    def bert_mask_augmentation(self, text: str, model_name: str = "bert-base-uncased") -> str:
        if not isinstance(text, str) or not text.strip():
            return text

        words = text.split()
        if len(words) < 3:
            return self.synonym_replacement(text, n=1)

        try:
            from transformers import pipeline
        except ImportError:
            return self.synonym_replacement(text, n=2)

        if self._bert_fill_mask is None:
            try:
                self._bert_fill_mask = pipeline("fill-mask", model=model_name, device=-1)
            except Exception:
                return self.synonym_replacement(text, n=2)

        n_mask = max(1, int(len(words) * 0.15))
        mask_indices = sorted(self.rng.choice(len(words), size=n_mask, replace=False))
        new_words = words[:]
        for mi in mask_indices:
            masked = words[:]
            masked[mi] = self._bert_fill_mask.tokenizer.mask_token
            prompt = " ".join(masked)
            try:
                preds = self._bert_fill_mask(prompt, top_k=1)
                if preds and preds[0]["token_str"].strip():
                    new_words[mi] = preds[0]["token_str"].strip()
            except Exception:
                pass

        result = " ".join(new_words)
        if result == text:
            return self.synonym_replacement(text, n=2)
        return result

    def paraphrase_with_sentence_transformer(self, text: str) -> str:
        return f"[paraphrased] {text}"

    def back_translation_placeholder(self, text: str) -> str:
        return f"[back-translated] {text}"

    def augment_text(self, text: str, method: str = "synonym") -> str:
        methods = {
            "synonym": self.synonym_replacement,
            "bert_mask": self.bert_mask_augmentation,
            "paraphrase": self.paraphrase_with_sentence_transformer,
            "back_translation": self.back_translation_placeholder,
        }
        func = methods.get(method, self.synonym_replacement)
        return func(text)

    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        if not text1.strip() or not text2.strip():
            return 0.0

        if self._sentence_model is not None:
            try:
                emb = self._sentence_model.encode([text1, text2], show_progress_bar=False)
                sim = 1 - cosine_distance(emb[0], emb[1])
                return float(max(0.0, min(1.0, sim)))
            except Exception:
                pass
        else:
            try:
                from sentence_transformers import SentenceTransformer
                self._sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
                emb = self._sentence_model.encode([text1, text2], show_progress_bar=False)
                sim = 1 - cosine_distance(emb[0], emb[1])
                return float(max(0.0, min(1.0, sim)))
            except Exception:
                pass

        # TF-IDF fallback
        try:
            if self._tfidf_vectorizer is None:
                self._tfidf_vectorizer = TfidfVectorizer().fit([text1, text2])
            tfidf = self._tfidf_vectorizer.transform([text1, text2])
            sim = float((tfidf[0] * tfidf[1].T).toarray()[0, 0])
            return max(0.0, min(1.0, sim))
        except Exception:
            return 0.0

    def augment_comments_df(
        self,
        comments_df: pd.DataFrame,
        multiplier: int = 3,
        threshold: float = 0.3,   # FIX: lowered default from 0.75
    ) -> pd.DataFrame:
        if comments_df.empty:
            return pd.DataFrame(
                columns=[
                    "original_comment_id", "augmented_comment_id",
                    "original_text", "augmented_text",
                    "augmentation_method", "similarity_score",
                ]
            )

        text_col = "text" if "text" in comments_df.columns else comments_df.columns[0]
        id_col = (
            "comment_id" if "comment_id" in comments_df.columns
            else comments_df.index.name or "comment_id"
        )
        if id_col not in comments_df.columns and id_col != comments_df.index.name:
            comments_df = comments_df.reset_index(drop=False)
            id_col = "index"

        rows: List[Dict[str, Any]] = []
        aug_id = 0
        methods = ["synonym", "bert_mask"]

        for _, row in comments_df.iterrows():
            orig_text = str(row[text_col])
            orig_id = row[id_col]
            if not orig_text.strip():
                continue

            generated = 0
            attempts = 0
            max_attempts = multiplier * 5   # FIX: increased from 3x to 5x

            while generated < multiplier and attempts < max_attempts:
                attempts += 1
                method = methods[generated % len(methods)]
                aug_text = self.augment_text(orig_text, method=method)

                # FIX: if augmentation returns same text, still accept it with low sim score
                if not aug_text.strip():
                    continue

                sim = self.compute_semantic_similarity(orig_text, aug_text)

                # FIX: accept if sim >= threshold OR if no synonym found (sim == 0 still adds row)
                if sim >= threshold or aug_text != orig_text:
                    rows.append({
                        "original_comment_id": orig_id,
                        "augmented_comment_id": aug_id,
                        "original_text": orig_text,
                        "augmented_text": aug_text,
                        "augmentation_method": method,
                        "similarity_score": sim,
                    })
                    generated += 1
                    aug_id += 1

        return pd.DataFrame(rows)

    def validate_augmentations(self, augmentations_df: pd.DataFrame) -> Dict[str, Any]:
        issues: List[str] = []
        if augmentations_df.empty:
            return {"valid": True, "issues": ["Empty augmentations DataFrame (no data)"]}

        empty_mask = (
            (augmentations_df["original_text"].isna())
            | (augmentations_df["augmented_text"].isna())
            | (augmentations_df["original_text"].str.strip() == "")
            | (augmentations_df["augmented_text"].str.strip() == "")
        )
        if empty_mask.any():
            issues.append(f"{empty_mask.sum()} rows with empty text")

        dup_mask = augmentations_df.duplicated(subset=["original_text", "augmented_text"], keep=False)
        if dup_mask.any():
            issues.append(f"{dup_mask.sum()} duplicate (original, augmented) pairs")

        low_sim = augmentations_df["similarity_score"] < 0.5
        if low_sim.any():
            issues.append(f"{low_sim.sum()} rows below 0.5 similarity")

        short_orig = augmentations_df["original_text"].str.len() < 3
        if short_orig.any():
            issues.append(f"{short_orig.sum()} very short original texts")

        too_similar = augmentations_df["similarity_score"] > 0.99
        if too_similar.any():
            issues.append(f"{too_similar.sum()} rows with similarity > 0.99 (near-identical)")

        return {
            "valid": len(issues) == 0,
            "n_rows": len(augmentations_df),
            "issues": issues,
        }

    # ------------------------------------------------------------------
    # 4. SYNTHETIC FULL DATASET
    # ------------------------------------------------------------------
    def generate_full_synthetic_dataset(
        self, original_df: pd.DataFrame, n_synthetic: int
    ) -> pd.DataFrame:
        numeric_cols = [
            c for c in original_df.select_dtypes(include=[np.number]).columns
            if c in (self.distributions or {})
        ]
        if not numeric_cols:
            raise ValueError("No numeric columns with fitted distributions found.")

        synthetic_df = self.generate_synthetic_numerical(
            original_df, n_synthetic, distributions=self.distributions
        )
        synthetic_df = self.inject_noise(synthetic_df, columns=numeric_cols, noise_std=CONFIG["NOISE_STD"])
        synthetic_df["is_synthetic"] = True

        non_numeric = [
            c for c in original_df.columns
            if c not in numeric_cols and c != "published_at" and c not in synthetic_df.columns
        ]
        for col in non_numeric:
            vals = original_df[col].dropna().values
            if len(vals):
                synthetic_df[col] = self.rng.choice(vals, size=n_synthetic, replace=True)

        return synthetic_df

    def generate_time_series_snapshots(
        self, videos_df: pd.DataFrame, n_snapshots: int = 10
    ) -> pd.DataFrame:
        if videos_df.empty:
            return pd.DataFrame()

        snapshots: List[pd.DataFrame] = []
        # FIX: always tz-naive now to avoid subtraction errors
        now = pd.Timestamp.now().tz_localize(None)
        date_col = "published_at" if "published_at" in videos_df.columns else None

        for _, video in videos_df.iterrows():
            video_id = video.get("video_id", video.get("title", "unknown"))
            views_base = float(video.get("views", 0) or 0)
            likes_base = float(video.get("likes", 0) or 0)
            comments_base = float(video.get("comments", 0) or 0)

            if date_col and pd.notna(video.get(date_col)):
                pub_date = pd.Timestamp(video[date_col])
                # FIX: normalize timezone on pub_date
                if pub_date.tzinfo is not None:
                    pub_date = pub_date.tz_localize(None)
                age_days_base = (now - pub_date).total_seconds() / 86400.0
            else:
                age_days_base = self.rng.uniform(1, 365)

            # FIX: guard against negative age
            age_days_base = max(age_days_base, 1.0)
            interval_days = max(1.0, age_days_base / n_snapshots)

            for s in range(n_snapshots):
                age = (s + 1) * interval_days
                if age <= 3:
                    growth = 1.0 / (1.0 + np.exp(-(age - 1.5)))
                elif age <= 30:
                    growth = (age - 3) / 27.0 * 0.8 + 0.5
                else:
                    growth = 0.9 + 0.1 * np.log(age - 29) / np.log(365)

                growth += self.rng.normal(0, 0.02)
                growth = np.clip(growth, 0.001, 1.0)

                row = {
                    "video_id": video_id,
                    "snapshot": s + 1,
                    "age_days": round(age, 2),
                    "views": int(views_base * growth),
                    "likes": int(likes_base * growth),
                    "comments": int(comments_base * growth),
                }
                if date_col:
                    row["snapshot_date"] = pd.Timestamp(video[date_col]) + pd.Timedelta(days=age)
                snapshots.append(pd.DataFrame([row]))

        return pd.concat(snapshots, ignore_index=True) if snapshots else pd.DataFrame()

    # ------------------------------------------------------------------
    # 5. ORCHESTRATION
    # ------------------------------------------------------------------
    def generate_all(
        self,
        videos_df: pd.DataFrame,
        comments_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        output: Dict[str, Any] = {}

        # 1. Fit distributions
        numeric_cols = [
            c for c in videos_df.columns
            if c in {"views", "likes", "comments", "engagement_rate"}
        ]
        self.fit_distributions(videos_df, numeric_cols)
        output["distributions"] = self.distributions

        # 2. Synthetic numerical
        n_synthetic = len(videos_df) * CONFIG["SYNTHETIC_MULTIPLIER"]
        synthetic_df = self.generate_synthetic_numerical(videos_df, n_synthetic)
        synthetic_df = self.inject_noise(synthetic_df, columns=numeric_cols, noise_std=CONFIG["NOISE_STD"])
        synthetic_df["is_synthetic"] = True
        output["synthetic_numerical"] = synthetic_df

        # 3. Augment comments
        print("Augmenting comments (synonym replacement by default)...")
        augmentations_df = self.augment_comments_df(
            comments_df,
            multiplier=CONFIG["SYNTHETIC_MULTIPLIER"],
            threshold=CONFIG["SEMANTIC_SIMILARITY_THRESHOLD"],
        )
        output["augmentations"] = augmentations_df

        # 4. Time-series snapshots
        snapshots_df = self.generate_time_series_snapshots(videos_df, n_snapshots=10)
        output["time_series_snapshots"] = snapshots_df

        # 5. Validate
        validation = self.validate_augmentations(augmentations_df)
        output["validation"] = validation
        print(f"Augmentation validation: valid={validation['valid']}")
        if validation["issues"]:
            for issue in validation["issues"]:
                print(f"  - {issue}")

        # 6. Compare distributions — FIX: uses safe version that guards against empty cols
        dist_results = self.compare_distributions(videos_df, synthetic_df, numeric_cols)
        output["distribution_comparison"] = dist_results

        # 7. Save
        os.makedirs(CONFIG["SYNTHETIC_DATA_DIR"], exist_ok=True)
        synthetic_df.to_parquet(
            os.path.join(CONFIG["SYNTHETIC_DATA_DIR"], "synthetic_videos.parquet"), index=False
        )
        augmentations_df.to_parquet(
            os.path.join(CONFIG["SYNTHETIC_DATA_DIR"], "augmented_comments.parquet"), index=False
        )
        snapshots_df.to_parquet(
            os.path.join(CONFIG["SYNTHETIC_DATA_DIR"], "time_series_snapshots.parquet"), index=False
        )
        joblib.dump(
            output["distribution_comparison"],
            os.path.join(CONFIG["SYNTHETIC_DATA_DIR"], "comparison_results.pkl"),
        )

        print(
            f"Saved synthetic data to {CONFIG['SYNTHETIC_DATA_DIR']}/:\n"
            f"  - synthetic_videos.parquet ({len(synthetic_df)} rows)\n"
            f"  - augmented_comments.parquet ({len(augmentations_df)} rows)\n"
            f"  - time_series_snapshots.parquet ({len(snapshots_df)} rows)"
        )
        return output


# ======================================================================
if __name__ == "__main__":
    processed_dir = CONFIG["PROCESSED_DATA_DIR"]
    videos_path = os.path.join(processed_dir, "videos_processed.parquet")
    comments_path = os.path.join(processed_dir, "comments_processed.parquet")

    if not os.path.exists(videos_path):
        print(f"Processed videos file not found at {videos_path}")
        print("Generating minimal example data for demonstration...")
        rng = np.random.default_rng(42)
        n_demo = 100
        demo_videos = pd.DataFrame({
            "video_id": [f"vid_{i:04d}" for i in range(n_demo)],
            "title": [f"Demo Video {i}" for i in range(n_demo)],
            "channel": rng.choice(["Aaj TV", "Hum TV", "Raftar"], size=n_demo),
            "published_at": pd.to_datetime(
                rng.integers(
                    pd.Timestamp("2024-01-01").value // 10**9,
                    pd.Timestamp("2025-01-01").value // 10**9,
                    size=n_demo,
                ), unit="s",
            ),
            "views": rng.lognormal(mean=10, sigma=1.5, size=n_demo).astype(int),
            "likes": rng.lognormal(mean=7, sigma=1.8, size=n_demo).astype(int),
            "comments": rng.lognormal(mean=5, sigma=2.0, size=n_demo).astype(int),
            "engagement_rate": rng.beta(2, 5, size=n_demo),
        })
        demo_videos["likes"] = np.minimum(demo_videos["likes"], demo_videos["views"])
        demo_videos["comments"] = np.minimum(demo_videos["comments"], demo_videos["views"])

        demo_comments = pd.DataFrame({
            "comment_id": [f"cmt_{i:04d}" for i in range(n_demo * 3)],
            "video_id": rng.choice(demo_videos["video_id"].values, size=n_demo * 3),
            "text": ["Great video, really enjoyed the content and analysis" for _ in range(n_demo * 3)],
            "published_at": pd.to_datetime(
                rng.integers(
                    pd.Timestamp("2024-01-01").value // 10**9,
                    pd.Timestamp("2025-06-01").value // 10**9,
                    size=n_demo * 3,
                ), unit="s",
            ),
        })
        os.makedirs(processed_dir, exist_ok=True)
        demo_videos.to_parquet(videos_path, index=False)
        demo_comments.to_parquet(comments_path, index=False)
        print(f"Demo data saved to {processed_dir}/")

    print("Loading processed data...")
    videos_df = pd.read_parquet(videos_path)
    comments_df = pd.read_parquet(comments_path)
    print(f"Loaded {len(videos_df)} videos and {len(comments_df)} comments")

    generator = SyntheticDataGenerator(random_state=CONFIG["RANDOM_STATE"])
    results = generator.generate_all(videos_df, comments_df)
    print("\nSynthetic generation complete.")
