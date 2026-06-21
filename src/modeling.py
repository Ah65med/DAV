"""
Modeling pipeline for digital media analytics.

Provides regression (engagement prediction), classification (content success
categorization), unsupervised clustering, temporal modeling, model
interpretability, and model persistence.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans

import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("modeling")

warnings.filterwarnings("ignore", category=FutureWarning)

CONFIG = {
    "RANDOM_STATE": 42,
    "TEST_SIZE": 0.2,
    "CV_FOLDS": 5,
    "SUCCESS_HIGH_PERCENTILE": 75,
    "SUCCESS_LOW_PERCENTILE": 25,
    "MODELS_DIR": "./outputs/models",
    "PROCESSED_DATA_DIR": "./data/processed",
}


class ModelingPipeline:
    """End-to-end modeling pipeline for digital media analytics.

    Parameters
    ----------
    random_state : int
        Seed for reproducible results (default 42).
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.scaler = StandardScaler()
        self._label_encoder = None
        self._trained_models: Dict[str, Any] = {}
        os.makedirs(CONFIG["MODELS_DIR"], exist_ok=True)

    # ==================================================================
    # REGRESSION: Engagement Prediction
    # ==================================================================

    def prepare_regression_data(
        self, features_df: pd.DataFrame, target_col: str = "engagement_rate"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train/test sets, filling missing values with column mean.

        Parameters
        ----------
        features_df : pd.DataFrame
            Feature matrix. Must contain *target_col*.
        target_col : str
            Name of the target column.

        Returns
        -------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        df = features_df.copy()
        y = df.pop(target_col)
        y = y.fillna(y.mean())
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())
        if len(df) < 5:
            logger.warning("Small dataset (%d rows). Results may be unreliable.", len(df))
        X_train, X_test, y_train, y_test = train_test_split(
            df[numeric_cols], y, test_size=CONFIG["TEST_SIZE"], random_state=self.random_state
        )
        logger.info("Regression data prepared: %d train, %d test", len(X_train), len(X_test))
        return X_train, X_test, y_train, y_test

    def train_linear_regression(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> LinearRegression:
        """Train a linear regression model.

        Returns
        -------
        LinearRegression
        """
        model = LinearRegression()
        model.fit(X_train, y_train)
        self._trained_models["linear_regression"] = model
        return model

    def train_random_forest_regressor(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> RandomForestRegressor:
        """Train a RandomForestRegressor with 100 estimators.

        Returns
        -------
        RandomForestRegressor
        """
        model = RandomForestRegressor(
            n_estimators=100, random_state=self.random_state, n_jobs=-1
        )
        model.fit(X_train, y_train)
        self._trained_models["random_forest_regressor"] = model
        return model

    def train_xgboost_regressor(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Any:
        """Train an XGBRegressor; falls back to RandomForest if xgboost unavailable.

        Returns
        -------
        XGBRegressor or RandomForestRegressor
        """
        try:
            import xgboost as xgb

            model = xgb.XGBRegressor(
                n_estimators=100,
                random_state=self.random_state,
                verbosity=0,
            )
            model.fit(X_train, y_train)
            self._trained_models["xgboost_regressor"] = model
            logger.info("XGBoost regressor trained successfully.")
            return model
        except ImportError:
            logger.warning("xgboost not available; falling back to RandomForestRegressor.")
            return self.train_random_forest_regressor(X_train, y_train)

    def evaluate_regression(
        self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str
    ) -> Dict[str, Any]:
        """Compute regression metrics: MAE, RMSE, R², MAPE.

        Returns
        -------
        dict
            Keys: ``mae``, ``rmse``, ``r2``, ``mape``.
        """
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        mape = _compute_mape(y_test, preds)
        logger.info("%s — MAE: %.4f, RMSE: %.4f, R²: %.4f, MAPE: %.2f%%", model_name, mae, rmse, r2, mape * 100)
        return {"model": model_name, "mae": mae, "rmse": rmse, "r2": r2, "mape": mape}

    def run_regression_pipeline(
        self, features_df: pd.DataFrame, target_col: str = "engagement_rate"
    ) -> Dict[str, Any]:
        """Train all regression models, evaluate, return results with feature importance.

        Returns
        -------
        dict
            ``train_test_data``, ``results`` (list of metric dicts),
            ``best_model``, ``feature_importance``.
        """
        if target_col not in features_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in features_df.")
        X_train, X_test, y_train, y_test = self.prepare_regression_data(features_df, target_col)

        models = {
            "LinearRegression": self.train_linear_regression(X_train, y_train),
            "RandomForestRegressor": self.train_random_forest_regressor(X_train, y_train),
            "XGBoostRegressor": self.train_xgboost_regressor(X_train, y_train),
        }

        results = []
        for name, model in models.items():
            results.append(self.evaluate_regression(model, X_test, y_test, name))

        best = max(results, key=lambda r: r["r2"])
        logger.info("Best regression model: %s (R²=%.4f)", best["model"], best["r2"])

        feature_importance = self.get_feature_importance(
            models["RandomForestRegressor"], X_train.columns.tolist()
        )

        return {
            "train_test_data": (X_train, X_test, y_train, y_test),
            "results": results,
            "best_model": best,
            "feature_importance": feature_importance,
        }

    # ==================================================================
    # CLASSIFICATION: Content Success Categorization
    # ==================================================================

    def create_success_labels(
        self, df: pd.DataFrame, metric_col: str = "engagement_rate"
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """Create success labels based on percentile thresholds.

        Labels:
          - High: top 25%
          - Medium: middle 50%
          - Low: bottom 25%

        Returns
        -------
        tuple
            (Series of labels, dict of threshold values)
        """
        high_thresh = df[metric_col].quantile(CONFIG["SUCCESS_HIGH_PERCENTILE"] / 100)
        low_thresh = df[metric_col].quantile(CONFIG["SUCCESS_LOW_PERCENTILE"] / 100)
        thresholds = {"high": high_thresh, "low": low_thresh}

        def _label(val: float) -> str:
            if val >= high_thresh:
                return "High"
            elif val <= low_thresh:
                return "Low"
            return "Medium"

        labels = df[metric_col].apply(_label)
        logger.info("Success labels created: %s", labels.value_counts().to_dict())
        return labels, thresholds

    def prepare_classification_data(
        self, features_df: pd.DataFrame, target_col: str = "success_label"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Stratified train/test split for classification.

        Returns
        -------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        df = features_df.copy()
        y = df.pop(target_col)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())
        X = df[numeric_cols]
        if len(X) < 5:
            logger.warning("Small dataset (%d rows) for classification.", len(X))
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=CONFIG["TEST_SIZE"], stratify=y, random_state=self.random_state
            )
        except ValueError:
            logger.warning("Stratified split failed (likely too few samples per class). Using regular split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=CONFIG["TEST_SIZE"], random_state=self.random_state
            )
        logger.info("Classification data prepared: %d train, %d test", len(X_train), len(X_test))
        return X_train, X_test, y_train, y_test

    def train_logistic_regression(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> LogisticRegression:
        """Train a multinomial logistic regression classifier.

        Returns
        -------
        LogisticRegression
        """
        model = LogisticRegression(
            multi_class="multinomial",
            max_iter=1000,
            random_state=self.random_state,
        )
        model.fit(X_train, y_train)
        self._trained_models["logistic_regression"] = model
        return model

    def train_random_forest_classifier(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> RandomForestClassifier:
        """Train a RandomForestClassifier.

        Returns
        -------
        RandomForestClassifier
        """
        model = RandomForestClassifier(
            n_estimators=100, random_state=self.random_state, n_jobs=-1
        )
        model.fit(X_train, y_train)
        self._trained_models["random_forest_classifier"] = model
        return model

    def train_xgboost_classifier(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Any:
        """Train an XGBClassifier; falls back to RandomForest if unavailable.

        Returns
        -------
        XGBClassifier or RandomForestClassifier
        """
        try:
            import xgboost as xgb

            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y_encoded = le.fit_transform(y_train)
            self._label_encoder = le

            model = xgb.XGBClassifier(
                n_estimators=100,
                random_state=self.random_state,
                verbosity=0,
                use_label_encoder=False,
            )
            model.fit(X_train, y_encoded)
            model._classes = le.classes_
            self._trained_models["xgboost_classifier"] = model
            logger.info("XGBoost classifier trained successfully.")
            return model
        except ImportError:
            logger.warning("xgboost not available; falling back to RandomForestClassifier.")
            return self.train_random_forest_classifier(X_train, y_train)

    def evaluate_classification(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
        label_encoder: Any = None,
    ) -> Dict[str, Any]:
        """Compute classification metrics: accuracy, per-class & weighted precision,
        recall, f1, plus confusion matrix.

        Returns
        -------
        dict
            ``model``, ``accuracy``, ``precision_per_class``, ``recall_per_class``,
            ``f1_per_class``, ``weighted_avg``, ``confusion_matrix``,
            ``classification_report``.
        """
        try:
            if label_encoder is not None:
                preds_encoded = model.predict(X_test)
                preds = label_encoder.inverse_transform(preds_encoded)
            elif hasattr(model, "_classes"):
                preds_encoded = model.predict(X_test)
                preds = model._classes[preds_encoded]
            else:
                preds = model.predict(X_test)

            if label_encoder is not None and hasattr(model, "_classes"):
                y_true = y_test.values
            else:
                y_true = y_test

            classes = sorted(set(y_true))
            accuracy = accuracy_score(y_true, preds)
            precision = precision_score(y_true, preds, average=None, labels=classes, zero_division=0)
            recall = recall_score(y_true, preds, average=None, labels=classes, zero_division=0)
            f1 = f1_score(y_true, preds, average=None, labels=classes, zero_division=0)
            weighted = {
                "precision": precision_score(y_true, preds, average="weighted", zero_division=0),
                "recall": recall_score(y_true, preds, average="weighted", zero_division=0),
                "f1": f1_score(y_true, preds, average="weighted", zero_division=0),
            }
            cm = confusion_matrix(y_true, preds, labels=classes)
            report = classification_report(y_true, preds, zero_division=0)

            logger.info("%s — Accuracy: %.4f", model_name, accuracy)
        except Exception as exc:
            logger.error("Evaluation failed for %s: %s", model_name, exc)
            return {"model": model_name, "error": str(exc)}

        return {
            "model": model_name,
            "accuracy": accuracy,
            "precision_per_class": dict(zip(classes, precision)),
            "recall_per_class": dict(zip(classes, recall)),
            "f1_per_class": dict(zip(classes, f1)),
            "weighted_avg": weighted,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

    def run_classification_pipeline(
        self, features_df: pd.DataFrame, metric_col: str = "engagement_rate"
    ) -> Dict[str, Any]:
        """Create success labels, train all classifiers, evaluate, return results.

        Returns
        -------
        dict
            ``labels``, ``thresholds``, ``train_test_data``,
            ``results``, ``best_model``.
        """
        labels, thresholds = self.create_success_labels(features_df, metric_col)
        df = features_df.copy()
        df["success_label"] = labels

        X_train, X_test, y_train, y_test = self.prepare_classification_data(df, "success_label")

        models = {
            "LogisticRegression": self.train_logistic_regression(X_train, y_train),
            "RandomForestClassifier": self.train_random_forest_classifier(X_train, y_train),
            "XGBoostClassifier": self.train_xgboost_classifier(X_train, y_train),
        }

        results = []
        for name, model in models.items():
            le = self._label_encoder if name == "XGBoostClassifier" else None
            results.append(self.evaluate_classification(model, X_test, y_test, name, label_encoder=le))

        valid = [r for r in results if "accuracy" in r]
        best = max(valid, key=lambda r: r["accuracy"]) if valid else None
        if best:
            logger.info("Best classifier: %s (Accuracy=%.4f)", best["model"], best["accuracy"])

        return {
            "labels": labels,
            "thresholds": thresholds,
            "train_test_data": (X_train, X_test, y_train, y_test),
            "results": results,
            "best_model": best,
        }

    # ==================================================================
    # UNSUPERVISED LEARNING: Clustering
    # ==================================================================

    def cluster_videos(
        self, features_df: pd.DataFrame, n_clusters: int = 5, method: str = "kmeans"
    ) -> np.ndarray:
        """Cluster videos using scaled numerical features.

        Parameters
        ----------
        features_df : pd.DataFrame
            Video features.
        n_clusters : int
            Number of clusters (for KMeans / Agglomerative).
        method : str
            ``'kmeans'``, ``'agglomerative'``, or ``'hdbscan'``.

        Returns
        -------
        np.ndarray
            Cluster labels.
        """
        numeric_df = features_df.select_dtypes(include=[np.number])
        for col in numeric_df.columns:
            if numeric_df[col].isna().any():
                numeric_df[col] = numeric_df[col].fillna(numeric_df[col].mean())
        X = self.scaler.fit_transform(numeric_df)

        if n_clusters > len(X):
            n_clusters = max(2, len(X) // 2)
            logger.warning("Reducing n_clusters to %d due to small dataset.", n_clusters)

        if method == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        elif method == "agglomerative":
            model = AgglomerativeClustering(n_clusters=n_clusters)
        elif method == "hdbscan":
            try:
                import hdbscan

                model = hdbscan.HDBSCAN(min_cluster_size=max(2, len(X) // max(n_clusters, 2)))
                labels = model.fit_predict(X)
                logger.info("HDBSCAN video clustering: %d clusters, %d noise points.",
                            len(set(labels)) - (1 if -1 in labels else 0),
                            (labels == -1).sum())
                return labels
            except ImportError:
                logger.warning("hdbscan not available; falling back to KMeans.")
                model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        labels = model.fit_predict(X)
        logger.info("%s video clustering: %d clusters.", method.capitalize(), len(set(labels)))
        return labels

    def cluster_comments(
        self, comment_embeddings: np.ndarray, n_clusters: int = 5
    ) -> np.ndarray:
        """Cluster comments using their embeddings (KMeans).

        Parameters
        ----------
        comment_embeddings : np.ndarray
            2D array of comment embedding vectors.
        n_clusters : int
            Number of clusters.

        Returns
        -------
        np.ndarray
            Cluster labels.
        """
        if comment_embeddings.ndim != 2:
            raise ValueError("comment_embeddings must be a 2D array.")

        X = self.scaler.fit_transform(comment_embeddings)
        if n_clusters > len(X):
            n_clusters = max(2, len(X) // 2)

        model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = model.fit_predict(X)
        logger.info("Comment clustering: %d clusters from %d embeddings.", len(set(labels)), len(X))
        return labels

    def evaluate_clustering(
        self, features: np.ndarray, labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute Silhouette, Davies-Bouldin, and Calinski-Harabasz scores.

        Returns
        -------
        dict
            ``silhouette_score``, ``davies_bouldin_index``,
            ``calinski_harabasz_score``.
        """
        unique_labels = set(labels)
        if len(unique_labels) <= 1:
            logger.warning("Only one cluster found; scores may be degenerate.")
            return {
                "silhouette_score": -1.0,
                "davies_bouldin_index": np.nan,
                "calinski_harabasz_score": 0.0,
            }
        if -1 in unique_labels:
            noise_mask = labels != -1
            features_clean = features[noise_mask]
            labels_clean = labels[noise_mask]
            if len(set(labels_clean)) <= 1:
                features_clean, labels_clean = features, labels
        else:
            features_clean, labels_clean = features, labels

        sil = silhouette_score(features_clean, labels_clean)
        db = davies_bouldin_score(features_clean, labels_clean)
        ch = calinski_harabasz_score(features_clean, labels_clean)
        logger.info("Clustering scores — Silhouette: %.4f, DB: %.4f, CH: %.2f", sil, db, ch)
        return {
            "silhouette_score": sil,
            "davies_bouldin_index": db,
            "calinski_harabasz_score": ch,
        }

    def interpret_clusters(
        self, features_df: pd.DataFrame, labels: np.ndarray, n_features: int = 5
    ) -> pd.DataFrame:
        """Compute mean feature values per cluster and identify distinguishing traits.

        Parameters
        ----------
        features_df : pd.DataFrame
            Feature matrix.
        labels : np.ndarray
            Cluster labels.
        n_features : int
            Number of top distinguishing features to report per cluster.

        Returns
        -------
        pd.DataFrame
            Cluster profiles with mean feature values.
        """
        numeric_df = features_df.select_dtypes(include=[np.number])
        numeric_df = numeric_df.copy()
        for col in numeric_df.columns:
            if numeric_df[col].isna().any():
                numeric_df[col] = numeric_df[col].fillna(numeric_df[col].mean())

        numeric_df["_cluster"] = labels
        profiles = numeric_df.groupby("_cluster").mean()

        overall_means = numeric_df.drop(columns=["_cluster"]).mean()

        observations = []
        for cluster_id, row in profiles.iterrows():
            deviations = row - overall_means
            top_high = deviations.nlargest(n_features).index.tolist()
            top_low = deviations.nsmallest(n_features).index.tolist()
            observations.append({
                "cluster": cluster_id,
                "size": (labels == cluster_id).sum(),
                "top_high_features": ", ".join(top_high),
                "top_low_features": ", ".join(top_low),
            })

        obs_df = pd.DataFrame(observations)
        logger.info("Cluster interpretation complete for %d clusters.", len(profiles))
        return profiles.join(obs_df.set_index("cluster"))

    def run_clustering_pipeline(
        self, features_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run KMeans and HDBSCAN clustering, evaluate both, return best result.

        Returns
        -------
        dict
            ``kmeans``, ``hdbscan``, ``best_method``, ``best_labels``,
            ``best_scores``, ``interpretation``.
        """
        numeric_df = features_df.select_dtypes(include=[np.number])
        for col in numeric_df.columns:
            if numeric_df[col].isna().any():
                numeric_df[col] = numeric_df[col].fillna(numeric_df[col].mean())
        X = numeric_df.values

        n_clusters = min(5, len(X) // 2) if len(X) > 2 else 2

        # KMeans
        km_labels = self.cluster_videos(features_df, n_clusters=n_clusters, method="kmeans")
        km_scores = self.evaluate_clustering(X, km_labels)

        # HDBSCAN
        try:
            hdb_labels = self.cluster_videos(features_df, n_clusters=n_clusters, method="hdbscan")
            hdb_scores = self.evaluate_clustering(X, hdb_labels)
        except Exception:
            logger.warning("HDBSCAN failed; using KMeans only.")
            hdb_labels = km_labels.copy()
            hdb_scores = {"silhouette_score": -1.0, "davies_bouldin_index": np.nan, "calinski_harabasz_score": 0.0}

        if km_scores.get("silhouette_score", -1) >= hdb_scores.get("silhouette_score", -1):
            best_method, best_labels, best_scores = "kmeans", km_labels, km_scores
        else:
            best_method, best_labels, best_scores = "hdbscan", hdb_labels, hdb_scores

        interpretation = self.interpret_clusters(features_df, best_labels)

        logger.info("Best clustering method: %s (Silhouette=%.4f)", best_method, best_scores["silhouette_score"])

        return {
            "kmeans": {"labels": km_labels, "scores": km_scores},
            "hdbscan": {"labels": hdb_labels, "scores": hdb_scores},
            "best_method": best_method,
            "best_labels": best_labels,
            "best_scores": best_scores,
            "interpretation": interpretation,
        }

    # ==================================================================
    # TEMPORAL MODELING
    # ==================================================================

    def train_temporal_model(
        self, time_series_data: pd.DataFrame
    ) -> Any:
        """Train a simplified temporal model using RandomForest with lag features.

        Creates lag features (t-1, t-2, t-3) and rolling statistics (mean,
        std over window 3) then fits a RandomForestRegressor.

        Parameters
        ----------
        time_series_data : pd.DataFrame
            Must have a datetime index or a numeric DataFrame where rows are
            sequential time steps.

        Returns
        -------
        RandomForestRegressor
            Trained model.
        """
        df = time_series_data.select_dtypes(include=[np.number]).copy()
        for col in df.columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())

        engineered = df.copy()
        for lag in [1, 2, 3]:
            for col in df.columns:
                engineered[f"{col}_lag{lag}"] = df[col].shift(lag)
        for col in df.columns:
            engineered[f"{col}_roll_mean3"] = df[col].rolling(window=3, min_periods=1).mean()
            engineered[f"{col}_roll_std3"] = df[col].rolling(window=3, min_periods=1).std()

        engineered = engineered.fillna(0)
        target_col = df.columns[0]

        X = engineered.drop(columns=df.columns.tolist()).values
        y = engineered[target_col].values

        model = RandomForestRegressor(
            n_estimators=100, random_state=self.random_state, n_jobs=-1
        )
        model.fit(X, y)
        self._feature_names_temporal = list(engineered.drop(columns=df.columns.tolist()).columns)
        logger.info("Temporal model trained on %d time steps with %d features.", len(X), X.shape[1])
        return model

    def predict_future_trend(
        self,
        model: Any,
        recent_data: pd.DataFrame,
        steps: int = 5,
    ) -> np.ndarray:
        """Predict engagement for the next *steps* time steps.

        Uses recent data rows to construct lag features autoregressively.

        Parameters
        ----------
        model : Any
            Trained temporal model (RandomForestRegressor).
        recent_data : pd.DataFrame
            Recent observations to seed the autoregressive prediction.
        steps : int
            Number of future steps to forecast.

        Returns
        -------
        np.ndarray
            Array of predicted values, shape (steps,).
        """
        df = recent_data.select_dtypes(include=[np.number]).copy()
        for col in df.columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())

        if len(df) < 4:
            logger.warning("Insufficient history (%d rows) for temporal prediction.", len(df))
            return np.full(steps, df.values[-1, 0])

        target_col = df.columns[0]
        predictions = []

        history = df.copy()
        for _step in range(steps):
            latest = history.iloc[-1:].copy()
            for lag in [1, 2, 3]:
                for col in history.columns:
                    shift_col = f"{col}_lag{lag}"
                    if lag <= len(history):
                        latest[shift_col] = history[col].iloc[-lag]
                    else:
                        latest[shift_col] = 0
            for col in history.columns:
                recent_window = history[col].iloc[-3:]
                latest[f"{col}_roll_mean3"] = recent_window.mean()
                latest[f"{col}_roll_std3"] = recent_window.std() if len(recent_window) > 1 else 0

            engineered_cols = [c for c in latest.columns if c.startswith(target_col)
                               or c.endswith(("_lag1", "_lag2", "_lag3", "_roll_mean3", "_roll_std3"))]
            feature_cols = [c for c in engineered_cols if c != target_col]

            feature_row = latest[feature_cols].fillna(0).values
            pred = model.predict(feature_row)[0]
            predictions.append(pred)

            new_row = history.iloc[-1:].copy()
            new_row[target_col] = pred
            history = pd.concat([history, new_row], ignore_index=True)

        return np.array(predictions)

    # ==================================================================
    # MODEL INTERPRETABILITY
    # ==================================================================

    def get_feature_importance(
        self, model: Any, feature_names: List[str], top_n: int = 15
    ) -> pd.DataFrame:
        """Return a DataFrame of feature importances.

        Uses ``feature_importances_`` for tree-based models, absolute
        coefficients for linear models.

        Parameters
        ----------
        model : Any
            Trained model.
        feature_names : list of str
            Feature column names.
        top_n : int
            Return only the top *top_n* features (default 15).

        Returns
        -------
        pd.DataFrame
            Columns: ``feature``, ``importance``, sorted descending.
        """
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim > 1:
                coef = coef[0]
            importances = np.abs(coef)
        else:
            logger.warning("Model has no feature_importances_ or coef_ attribute.")
            return pd.DataFrame(columns=["feature", "importance"])

        if len(importances) != len(feature_names):
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        df = pd.DataFrame({"feature": feature_names, "importance": importances})
        df = df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
        return df

    def explain_with_shap(
        self, model: Any, X_sample: pd.DataFrame, feature_names: List[str]
    ) -> Optional[Any]:
        """Attempt to compute SHAP values; falls back gracefully if shap unavailable.

        Parameters
        ----------
        model : Any
            Trained model.
        X_sample : pd.DataFrame
            Sample of feature data to explain.
        feature_names : list of str
            Feature column names.

        Returns
        -------
        shap.Explanation or None
            SHAP explanation object, or None if unavailable.
        """
        try:
            import shap

            if isinstance(model, (RandomForestRegressor, RandomForestClassifier)):
                explainer = shap.TreeExplainer(model)
            else:
                background = X_sample.sample(
                    n=min(50, len(X_sample)), random_state=self.random_state
                )
                explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer(X_sample[:min(100, len(X_sample))])
            logger.info("SHAP explanation computed successfully.")
            return shap_values
        except ImportError:
            logger.warning("shap package not available. Install with: pip install shap")
            return None
        except Exception as exc:
            logger.warning("SHAP computation failed: %s", exc)
            return None

    # ==================================================================
    # MODEL PERSISTENCE
    # ==================================================================

    def save_model(self, model: Any, filename: str) -> str:
        """Save a model to disk using joblib.

        Parameters
        ----------
        model : Any
            Trained model object.
        filename : str
            Output filename (basename only).

        Returns
        -------
        str
            Full path to the saved model.
        """
        filepath = os.path.join(CONFIG["MODELS_DIR"], filename)
        joblib.dump(model, filepath)
        logger.info("Model saved to %s", filepath)
        return filepath

    def load_model(self, filename: str) -> Any:
        """Load a model from disk using joblib.

        Parameters
        ----------
        filename : str
            Filename (basename only).

        Returns
        -------
        Any
            Loaded model object.
        """
        filepath = os.path.join(CONFIG["MODELS_DIR"], filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        model = joblib.load(filepath)
        logger.info("Model loaded from %s", filepath)
        return model

    # ==================================================================
    # FULL PIPELINE
    # ==================================================================

    def run_full_modeling_pipeline(
        self, features_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Orchestrate regression, classification, and clustering in one pass.

        Parameters
        ----------
        features_df : pd.DataFrame
            Feature matrix must contain ``engagement_rate``.

        Returns
        -------
        dict
            ``regression``, ``classification``, ``clustering``.
        """
        if "engagement_rate" not in features_df.columns:
            raise ValueError("features_df must contain 'engagement_rate' column.")

        print("=" * 60)
        print("Starting full modeling pipeline\n")

        print("[1/3] Running regression pipeline...")
        regression_results = self.run_regression_pipeline(features_df, target_col="engagement_rate")
        print(f"  Best model: {regression_results['best_model']['model']} (R²={regression_results['best_model']['r2']:.4f})")

        print("\n[2/3] Running classification pipeline...")
        classification_results = self.run_classification_pipeline(features_df, metric_col="engagement_rate")
        if classification_results["best_model"]:
            print(f"  Best model: {classification_results['best_model']['model']} (Accuracy={classification_results['best_model']['accuracy']:.4f})")
        else:
            print("  Classification failed; no valid results.")

        print("\n[3/3] Running clustering pipeline...")
        clustering_results = self.run_clustering_pipeline(features_df)
        print(f"  Best method: {clustering_results['best_method']} (Silhouette={clustering_results['best_scores']['silhouette_score']:.4f})")

        print("\nPipeline complete.")
        print("=" * 60)

        return {
            "regression": regression_results,
            "classification": classification_results,
            "clustering": clustering_results,
        }

    def generate_modeling_report(self, results: Dict[str, Any]) -> str:
        """Generate a text summary of all modeling results.

        Parameters
        ----------
        results : dict
            Output from ``run_full_modeling_pipeline``.

        Returns
        -------
        str
            Formatted report string.
        """
        lines = [
            "=" * 60,
            "DIGITAL MEDIA ANALYTICS - MODELING REPORT",
            "=" * 60,
            "",
        ]

        reg = results.get("regression", {})
        if reg:
            lines.append("--- REGRESSION (Engagement Prediction) ---")
            for r in reg.get("results", []):
                lines.append(f"  {r['model']:30s}  MAE={r['mae']:.4f}  RMSE={r['rmse']:.4f}  R²={r['r2']:.4f}  MAPE={r['mape']*100:.2f}%")
            best = reg.get("best_model", {})
            if best:
                lines.append(f"  Best: {best['model']} (R²={best['r2']:.4f})")
            fi = reg.get("feature_importance")
            if fi is not None and not fi.empty:
                lines.append("  Top features:")
                for _, row in fi.iterrows():
                    lines.append(f"    {row['feature']:40s} {row['importance']:.4f}")
            lines.append("")

        clf = results.get("classification", {})
        if clf:
            lines.append("--- CLASSIFICATION (Content Success) ---")
            lines.append(f"  Thresholds: High ≥ {clf.get('thresholds', {}).get('high', 'N/A'):.4f}, Low ≤ {clf.get('thresholds', {}).get('low', 'N/A'):.4f}")
            for r in clf.get("results", []):
                if "accuracy" in r:
                    lines.append(f"  {r['model']:30s}  Accuracy={r['accuracy']:.4f}  Weighted F1={r['weighted_avg']['f1']:.4f}")
            best = clf.get("best_model", {})
            if best:
                lines.append(f"  Best: {best['model']} (Accuracy={best['accuracy']:.4f})")
            lines.append("")

        clust = results.get("clustering", {})
        if clust:
            lines.append("--- CLUSTERING ---")
            lines.append(f"  Best method: {clust.get('best_method', 'N/A')}")
            scores = clust.get("best_scores", {})
            lines.append(f"  Silhouette: {scores.get('silhouette_score', 'N/A'):.4f}")
            lines.append(f"  Davies-Bouldin: {scores.get('davies_bouldin_index', 'N/A'):.4f}")
            lines.append(f"  Calinski-Harabasz: {scores.get('calinski_harabasz_score', 'N/A'):.2f}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error, handling zeros in y_true."""
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


# ======================================================================
# __main__
# ======================================================================

if __name__ == "__main__":
    data_dir = CONFIG["PROCESSED_DATA_DIR"]
    models_dir = CONFIG["MODELS_DIR"]
    os.makedirs(models_dir, exist_ok=True)

    features_path = os.path.join(data_dir, "videos_processed.parquet")
    if not os.path.exists(features_path):
        logger.warning("Processed features file not found at %s. Checking for CSV fallback.", features_path)
        features_path = os.path.join(data_dir, "videos_processed.csv")
    if not os.path.exists(features_path):
        logger.error("No processed features file found in %s. Run preprocessing first.", data_dir)
        raise SystemExit(1)

    logger.info("Loading features from %s", features_path)
    if features_path.endswith(".parquet"):
        features_df = pd.read_parquet(features_path)
    else:
        features_df = pd.read_csv(features_path)

    if "engagement_rate" not in features_df.columns:
        logger.error("'engagement_rate' column missing. Ensure preprocessing generated engagement metrics.")
        raise SystemExit(1)

    logger.info("Loaded features: %s", features_df.shape)

    pipeline = ModelingPipeline(random_state=CONFIG["RANDOM_STATE"])
    results = pipeline.run_full_modeling_pipeline(features_df)

    report = pipeline.generate_modeling_report(results)
    print("\n" + report)

    report_path = os.path.join(CONFIG["MODELS_DIR"], "modeling_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    logger.info("Modeling report saved to %s", report_path)

    for model_name in ["random_forest_regressor", "random_forest_classifier"]:
        if model_name in pipeline._trained_models:
            pipeline.save_model(pipeline._trained_models[model_name], f"{model_name}.joblib")

    results_path = os.path.join(CONFIG["MODELS_DIR"], "modeling_results.joblib")
    serializable = {
        "regression": {
            "results": results["regression"]["results"],
            "best_model": results["regression"]["best_model"],
        },
        "classification": {
            "thresholds": results["classification"]["thresholds"],
            "results": results["classification"]["results"],
            "best_model": results["classification"]["best_model"],
        },
        "clustering": {
            "best_method": results["clustering"]["best_method"],
            "best_scores": results["clustering"]["best_scores"],
        },
    }
    joblib.dump(serializable, results_path)
    logger.info("Modeling results saved to %s", results_path)

    print("\nDone. Models and results saved.")
