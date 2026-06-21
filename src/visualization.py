"""
Comprehensive visualization engine for digital media analytics.

Provides statistical, temporal, sentiment/NLP, model, clustering, graph,
interactive Plotly, and dashboard visualizations. All plots are
publication-quality with consistent channel color palettes.
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG: Dict[str, Any] = {
    "RANDOM_STATE": 42,
    "FIGURES_DIR": "./outputs/figures",
    "GRAPHS_DIR": "./outputs/graphs",
    "PROCESSED_DATA_DIR": "./data/processed",
}

CHANNEL_COLORS: Dict[str, str] = {
    "Aaj TV (Aaj News)": "#1f77b4",
    "Hum TV": "#ff7f0e",
    "Raftar": "#2ca02c",
}

CHANNEL_COLORS_FALLBACK: List[str] = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first matching column name (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower:
            return lower[key]
    return None


def _resolve_many(
    df: pd.DataFrame, mapping: Dict[str, List[str]]
) -> Dict[str, str]:
    """Resolve a dict of {logical_name: [candidate_column_names]}."""
    resolved: Dict[str, str] = {}
    for logical, candidates in mapping.items():
        col = _resolve_col(df, candidates)
        if col:
            resolved[logical] = col
    return resolved


def _get_channel_colors(channel_names: List[str]) -> Dict[str, str]:
    """Return a colour dict for the given channel names."""
    colors: Dict[str, str] = {}
    for i, name in enumerate(channel_names):
        if name in CHANNEL_COLORS:
            colors[name] = CHANNEL_COLORS[name]
        else:
            colors[name] = CHANNEL_COLORS_FALLBACK[
                i % len(CHANNEL_COLORS_FALLBACK)
            ]
    return colors


def _sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._- " else "_" for c in name)


# ---------------------------------------------------------------------------
# VisualizationEngine
# ---------------------------------------------------------------------------


class VisualizationEngine:
    """Publication-quality visualization engine for digital media analytics.

    Parameters
    ----------
    figsize : tuple
        Default figure size (width, height) in inches.
    dpi : int
        Default figure resolution.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 6), dpi: int = 100) -> None:
        self.figsize = figsize
        self.dpi = dpi
        sns.set_style("whitegrid")
        matplotlib.rcParams.update(
            {
                "figure.figsize": figsize,
                "figure.dpi": dpi,
                "font.size": 11,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
            }
        )

        os.makedirs(CONFIG["FIGURES_DIR"], exist_ok=True)
        os.makedirs(CONFIG["GRAPHS_DIR"], exist_ok=True)

    # ==================================================================
    # Utility
    # ==================================================================

    def save_figure(
        self,
        fig: plt.Figure,
        filename: str,
        dpi: int = 150,
        subdir: str = "figures",
    ) -> str:
        """Save a matplotlib figure with tight_layout."""
        base = (
            CONFIG["FIGURES_DIR"]
            if subdir == "figures"
            else CONFIG["GRAPHS_DIR"]
        )
        path = os.path.join(base, filename)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Saved figure → %s", path)
        return path

    def _get_channel_col(self, df: pd.DataFrame) -> Optional[str]:
        return _resolve_col(
            df,
            ["channel_name", "channel_title", "channelName", "channel"],
        )

    def _channel_names(self, df: pd.DataFrame) -> List[str]:
        col = self._get_channel_col(df)
        if col is None:
            return []
        return sorted(df[col].dropna().unique().tolist())

    # ==================================================================
    # 1. STATISTICAL VISUALIZATIONS
    # ==================================================================

    def plot_views_distribution(
        self, videos_df: pd.DataFrame, save: bool = True
    ) -> Optional[plt.Figure]:
        """Histogram with KDE overlay of views, coloured by channel."""
        cols = _resolve_many(
            videos_df,
            {
                "views": ["views", "view_count"],
                "channel": ["channel_name", "channel_title", "channel"],
            },
        )
        if "views" not in cols or "channel" not in cols:
            logger.warning("plot_views_distribution: missing required columns.")
            return None

        v_col = cols["views"]
        ch_col = cols["channel"]
        channels = self._channel_names(videos_df)
        ch_colors = _get_channel_colors(channels)

        fig, axes = plt.subplots(
            1, len(channels), figsize=(6 * len(channels), 5), sharey=True
        )
        if len(channels) == 1:
            axes = [axes]

        for ax, ch in zip(axes, channels):
            data = videos_df[videos_df[ch_col] == ch][v_col].dropna()
            data_log = np.log1p(data[data > 0])
            if len(data_log) > 1:
                sns.histplot(
                    data_log,
                    bins=40,
                    kde=True,
                    color=ch_colors.get(ch, "#1f77b4"),
                    edgecolor="white",
                    alpha=0.6,
                    ax=ax,
                )
            ax.set_title(f"{ch}")
            ax.set_xlabel("log(Views + 1)")
            ax.set_ylabel("Frequency")

        fig.suptitle("Views Distribution by Channel", fontsize=15, y=1.02)
        if save:
            self.save_figure(fig, "views_distribution.png")
        return fig

    def plot_likes_distribution(
        self, videos_df: pd.DataFrame, save: bool = True
    ) -> Optional[plt.Figure]:
        """Boxplot of likes by channel."""
        cols = _resolve_many(
            videos_df,
            {
                "likes": ["likes", "like_count"],
                "channel": ["channel_name", "channel_title", "channel"],
            },
        )
        if "likes" not in cols:
            logger.warning("plot_likes_distribution: missing likes column.")
            return None

        fig, ax = plt.subplots(figsize=self.figsize)
        ch_col = cols.get("channel")
        if ch_col:
            order = sorted(videos_df[ch_col].dropna().unique())
            palette = _get_channel_colors(order)
            sns.boxplot(
                x=ch_col,
                y=cols["likes"],
                data=videos_df,
                palette=palette,
                order=order,
                ax=ax,
            )
        else:
            sns.boxplot(y=cols["likes"], data=videos_df, ax=ax)

        ax.set_title("Likes Distribution by Channel")
        ax.set_ylabel("Likes")
        ax.set_yscale("log")
        if save:
            self.save_figure(fig, "likes_distribution.png")
        return fig

    def plot_comments_distribution(
        self, videos_df: pd.DataFrame, save: bool = True
    ) -> Optional[plt.Figure]:
        """Boxplot of comments by channel."""
        cols = _resolve_many(
            videos_df,
            {
                "comments": ["comments", "comment_count"],
                "channel": ["channel_name", "channel_title", "channel"],
            },
        )
        if "comments" not in cols:
            logger.warning("plot_comments_distribution: missing comments column.")
            return None

        fig, ax = plt.subplots(figsize=self.figsize)
        ch_col = cols.get("channel")
        if ch_col:
            order = sorted(videos_df[ch_col].dropna().unique())
            palette = _get_channel_colors(order)
            sns.boxplot(
                x=ch_col,
                y=cols["comments"],
                data=videos_df,
                palette=palette,
                order=order,
                ax=ax,
            )
        else:
            sns.boxplot(y=cols["comments"], data=videos_df, ax=ax)

        ax.set_title("Comments Distribution by Channel")
        ax.set_ylabel("Comments")
        ax.set_yscale("log")
        if save:
            self.save_figure(fig, "comments_distribution.png")
        return fig

    def plot_engagement_comparison(
        self, videos_df: pd.DataFrame, save: bool = True
    ) -> Optional[plt.Figure]:
        """Bar chart of mean engagement_rate per channel with error bars."""
        cols = _resolve_many(
            videos_df,
            {
                "engagement_rate": [
                    "engagement_rate",
                    "engagementRate",
                ],
                "channel": ["channel_name", "channel_title", "channel"],
            },
        )
        er_col = cols.get("engagement_rate")
        if er_col is None:
            # Try to derive from likes+comments / views
            l_col = _resolve_col(videos_df, ["likes", "like_count"])
            c_col = _resolve_col(videos_df, ["comments", "comment_count"])
            v_col = _resolve_col(videos_df, ["views", "view_count"])
            if l_col and v_col and c_col:
                videos_df = videos_df.copy()
                videos_df["_engagement_rate"] = np.where(
                    videos_df[v_col] > 0,
                    (videos_df[l_col] + videos_df[c_col]) / videos_df[v_col],
                    0.0,
                )
                er_col = "_engagement_rate"

        if er_col is None:
            logger.warning("plot_engagement_comparison: missing engagement_rate.")
            return None

        ch_col = cols.get("channel")
        fig, ax = plt.subplots(figsize=self.figsize)

        if ch_col:
            order = sorted(videos_df[ch_col].dropna().unique())
            stats_df = (
                videos_df.groupby(ch_col)[er_col]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            stats_df["ci"] = (
                1.96 * stats_df["std"] / np.sqrt(stats_df["count"])
            )
            palette = _get_channel_colors(order)
            bars = ax.bar(
                stats_df[ch_col],
                stats_df["mean"],
                yerr=stats_df["ci"],
                color=[palette.get(ch, "#1f77b4") for ch in stats_df[ch_col]],
                capsize=5,
                edgecolor="black",
                linewidth=0.8,
            )
            for bar, mean_val in zip(bars, stats_df["mean"]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + stats_df["ci"].iloc[0] * 0.1,
                    f"{mean_val:.3f}",
                    ha="center",
                    fontsize=9,
                )
        else:
            mean_val = videos_df[er_col].mean()
            std_val = videos_df[er_col].std()
            n = len(videos_df)
            ci = 1.96 * std_val / np.sqrt(n) if n > 0 else 0
            ax.bar(["All"], [mean_val], yerr=[ci], capsize=5, color="#1f77b4")
            ax.text(0, mean_val + ci * 0.1, f"{mean_val:.3f}", ha="center")

        ax.set_title("Mean Engagement Rate by Channel")
        ax.set_ylabel("Engagement Rate")
        ax.set_ylim(bottom=0)
        if save:
            self.save_figure(fig, "engagement_comparison.png")
        return fig

    def plot_correlation_heatmap(
        self,
        videos_df: pd.DataFrame,
        numerical_cols: Optional[List[str]] = None,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Annotated seaborn heatmap of the correlation matrix."""
        if numerical_cols is None:
            candidates = [
                "views",
                "view_count",
                "likes",
                "like_count",
                "comments",
                "comment_count",
                "engagement_rate",
                "like_rate",
                "comment_rate",
                "virality_score",
                "video_age_days",
                "duration_seconds",
            ]
            numerical_cols = [c for c in candidates if c in videos_df.columns]
            # Also include any numeric cols not in candidates
            extra_num = [
                c
                for c in videos_df.select_dtypes(include=[np.number]).columns
                if c not in numerical_cols
                and not c.endswith("_missing")
                and c not in {"hour", "day_of_week", "month", "year", "is_weekend", "outlier_label"}
            ]
            numerical_cols.extend(extra_num)

        available = [c for c in numerical_cols if c in videos_df.columns]
        if len(available) < 2:
            logger.warning("plot_correlation_heatmap: need at least 2 numeric columns.")
            return None

        corr = videos_df[available].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            mask=mask,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
        ax.set_title("Correlation Matrix of Numerical Features")
        if save:
            self.save_figure(fig, "correlation_heatmap.png")
        return fig

    def plot_top_performing_videos(
        self,
        videos_df: pd.DataFrame,
        metric: str = "views",
        n: int = 15,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Horizontal bar chart of top N videos by metric."""
        metric_col = _resolve_col(
            videos_df,
            [metric, f"{metric}_count", metric.lower()],
        )
        title_col = _resolve_col(videos_df, ["title", "video_title"])
        ch_col = self._get_channel_col(videos_df)

        if metric_col is None:
            metric_col = _resolve_col(videos_df, ["views", "view_count"])
        if metric_col is None:
            logger.warning("plot_top_performing_videos: no metric column found.")
            return None

        df = videos_df.sort_values(metric_col, ascending=False).head(n).copy()
        if title_col:
            df["_label"] = df[title_col].str[:60]
        else:
            df["_label"] = df.index.astype(str)

        fig, ax = plt.subplots(figsize=(10, max(6, n * 0.35)))
        colors = None
        if ch_col and ch_col in df.columns:
            channels = df[ch_col].unique()
            palette = _get_channel_colors(list(channels))
            colors = [palette.get(ch, "#1f77b4") for ch in df[ch_col]]

        ax.barh(
            df["_label"][::-1],
            df[metric_col].values[::-1],
            color=colors[::-1] if colors else "#1f77b4",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_xlabel(metric_col.replace("_", " ").title())
        ax.set_title(f"Top {n} Videos by {metric.replace('_', ' ').title()}")
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
        )
        if save:
            self.save_figure(fig, f"top_{n}_{metric}_videos.png")
        return fig

    def plot_scatter_matrix(
        self,
        videos_df: pd.DataFrame,
        cols: Optional[List[str]] = None,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Pairplot of numerical columns coloured by channel."""
        ch_col = self._get_channel_col(videos_df)

        if cols is None:
            cols = [
                "views",
                "view_count",
                "likes",
                "like_count",
                "comments",
                "comment_count",
                "engagement_rate",
            ]
        resolved = []
        for c in cols:
            r = _resolve_col(videos_df, [c])
            if r:
                resolved.append(r)

        # deduplicate keeping order
        seen = set()
        resolved_unique = []
        for c in resolved:
            if c not in seen:
                seen.add(c)
                resolved_unique.append(c)

        if len(resolved_unique) < 2:
            logger.warning("plot_scatter_matrix: need at least 2 columns.")
            return None

        plot_data = videos_df[resolved_unique + ([ch_col] if ch_col else [])].dropna()
        plot_data = plot_data.apply(lambda s: np.log1p(s) if s.name in resolved_unique and s.dtype in ["float64", "int64"] else s)

        g = sns.pairplot(
            plot_data,
            vars=resolved_unique,
            hue=ch_col,
            diag_kind="kde",
            plot_kws={"alpha": 0.5, "s": 15},
            height=2.5,
        )
        g.fig.suptitle("Scatter Matrix by Channel", y=1.02, fontsize=15)
        if save:
            self.save_figure(g.fig, "scatter_matrix.png")
        return g.fig

    def plot_anomaly_scatter(
        self,
        videos_df: pd.DataFrame,
        x: str = "views",
        y: str = "engagement_rate",
        outlier_col: str = "outlier_label",
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Scatter plot highlighting outliers."""
        x_col = _resolve_col(videos_df, [x])
        y_col = _resolve_col(videos_df, [y])
        oc_col = _resolve_col(videos_df, [outlier_col])

        if x_col is None or y_col is None:
            logger.warning("plot_anomaly_scatter: missing x or y columns.")
            return None

        fig, ax = plt.subplots(figsize=self.figsize)

        normal = videos_df
        outliers = pd.DataFrame()
        if oc_col:
            mask_normal = videos_df[oc_col] == 0
            normal = videos_df[mask_normal]
            outliers = videos_df[~mask_normal]

        ax.scatter(
            normal[x_col],
            normal[y_col],
            alpha=0.4,
            s=15,
            c="steelblue",
            label="Normal",
            edgecolors="none",
        )
        if len(outliers) > 0:
            ax.scatter(
                outliers[x_col],
                outliers[y_col],
                alpha=0.8,
                s=40,
                c="crimson",
                marker="x",
                linewidths=1.5,
                label=f"Outlier (n={len(outliers)})",
            )

        ax.set_xlabel(x.replace("_", " ").title())
        ax.set_ylabel(y.replace("_", " ").title())
        ax.set_title(f"Anomaly Detection: {x} vs {y}")
        ax.legend(loc="best")
        ax.set_xscale("log")

        if save:
            self.save_figure(fig, "anomaly_scatter.png")
        return fig

    # ==================================================================
    # 2. TEMPORAL VISUALIZATIONS
    # ==================================================================

    def plot_engagement_timeline(
        self, videos_df: pd.DataFrame, save: bool = True
    ) -> Optional[plt.Figure]:
        """Line plot of engagement over time per channel."""
        ts_col = _resolve_col(videos_df, ["published_at", "publishedAt"])
        er_col = _resolve_col(videos_df, ["engagement_rate"])
        ch_col = self._get_channel_col(videos_df)

        if ts_col is None:
            logger.warning("plot_engagement_timeline: missing timestamp column.")
            return None

        df = videos_df.copy()
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col]).sort_values(ts_col)
        df["_period"] = df[ts_col].dt.to_period("W").astype(str)

        if er_col is None:
            v_col = _resolve_col(df, ["views", "view_count"])
            l_col = _resolve_col(df, ["likes", "like_count"])
            c_col = _resolve_col(df, ["comments", "comment_count"])
            if v_col and l_col and c_col:
                df[er_col] = np.where(
                    df[v_col] > 0,
                    (df[l_col] + df[c_col]) / df[v_col],
                    0.0,
                )
                er_col = "engagement_rate"

        if er_col is None:
            logger.warning("plot_engagement_timeline: missing engagement rate.")
            return None

        fig, ax = plt.subplots(figsize=self.figsize)

        if ch_col:
            channels = sorted(df[ch_col].dropna().unique())
            palette = _get_channel_colors(channels)
            for ch in channels:
                sub = df[df[ch_col] == ch]
                agg = sub.groupby("_period")[er_col].mean().reset_index()
                ax.plot(
                    range(len(agg)),
                    agg[er_col],
                    marker="o",
                    markersize=3,
                    linewidth=1.5,
                    label=ch,
                    color=palette.get(ch, "#1f77b4"),
                )
                # Show every Nth tick label to avoid clutter
                step = max(1, len(agg) // 10)
                tick_positions = list(range(0, len(agg), step))
                tick_labels = [agg["_period"].iloc[i] for i in tick_positions]
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
            ax.legend(loc="best")
        else:
            agg = df.groupby("_period")[er_col].mean().reset_index()
            ax.plot(
                range(len(agg)),
                agg[er_col],
                marker="o",
                markersize=3,
                linewidth=1.5,
                color="#1f77b4",
            )
            step = max(1, len(agg) // 10)
            tick_positions = list(range(0, len(agg), step))
            tick_labels = [agg["_period"].iloc[i] for i in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

        ax.set_title("Engagement Rate Timeline by Channel")
        ax.set_ylabel("Mean Engagement Rate")
        ax.set_xlabel("Week")
        if save:
            self.save_figure(fig, "engagement_timeline.png")
        return fig

    def plot_posting_time_heatmap(
        self, videos_df: pd.DataFrame, save: bool = True
    ) -> Optional[plt.Figure]:
        """Heatmap of day_of_week vs hour — video count."""
        hour_col = _resolve_col(videos_df, ["hour"])
        dow_col = _resolve_col(videos_df, ["day_of_week"])

        if hour_col is None or dow_col is None:
            ts_col = _resolve_col(videos_df, ["published_at", "publishedAt"])
            if ts_col is not None:
                df = videos_df.copy()
                df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
                df["hour"] = df[ts_col].dt.hour
                df["day_of_week"] = df[ts_col].dt.dayofweek
                hour_col = "hour"
                dow_col = "day_of_week"
            else:
                logger.warning("plot_posting_time_heatmap: missing temporal columns.")
                return None

        pivot = (
            videos_df.groupby([dow_col, hour_col])
            .size()
            .unstack(fill_value=0)
        )

        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        # Align day indices
        for d in range(7):
            if d not in pivot.index:
                pivot.loc[d] = 0
        pivot = pivot.sort_index()

        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(
            pivot,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "Video Count"},
        )
        ax.set_yticklabels(
            [day_labels[i] if i < len(day_labels) else str(i) for i in pivot.index],
            rotation=0,
        )
        ax.set_title("Posting Activity: Day of Week vs Hour")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Day of Week")
        if save:
            self.save_figure(fig, "posting_time_heatmap.png")
        return fig

    def plot_day_of_week_effect(
        self, videos_df: pd.DataFrame, save: bool = True
    ) -> Optional[plt.Figure]:
        """Bar chart of average engagement by day of week per channel."""
        ch_col = self._get_channel_col(videos_df)
        dow_col = _resolve_col(videos_df, ["day_of_week"])
        er_col = _resolve_col(videos_df, ["engagement_rate"])

        if dow_col is None:
            ts_col = _resolve_col(videos_df, ["published_at", "publishedAt"])
            if ts_col is not None:
                df = videos_df.copy()
                df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
                df["day_of_week"] = df[ts_col].dt.dayofweek
                dow_col = "day_of_week"
            else:
                logger.warning("plot_day_of_week_effect: missing day_of_week.")
                return None

        if er_col is None:
            v_col = _resolve_col(videos_df, ["views", "view_count"])
            l_col = _resolve_col(videos_df, ["likes", "like_count"])
            c_col = _resolve_col(videos_df, ["comments", "comment_count"])
            if v_col and l_col and c_col:
                videos_df = videos_df.copy()
                videos_df["_engagement_rate"] = np.where(
                    videos_df[v_col] > 0,
                    (videos_df[l_col] + videos_df[c_col]) / videos_df[v_col],
                    0.0,
                )
                er_col = "_engagement_rate"

        if er_col is None:
            logger.warning("plot_day_of_week_effect: missing engagement rate.")
            return None

        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        fig, ax = plt.subplots(figsize=self.figsize)

        if ch_col:
            channels = sorted(videos_df[ch_col].dropna().unique())
            palette = _get_channel_colors(channels)
            x = np.arange(7)
            width = 0.8 / len(channels)
            for i, ch in enumerate(channels):
                sub = videos_df[videos_df[ch_col] == ch]
                means = sub.groupby(dow_col)[er_col].mean()
                vals = [means.get(d, 0) for d in range(7)]
                ax.bar(
                    x + i * width - 0.4 + width / 2,
                    vals,
                    width,
                    label=ch,
                    color=palette.get(ch, "#1f77b4"),
                    edgecolor="black",
                    linewidth=0.5,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(day_labels)
            ax.legend(loc="best")
        else:
            means = videos_df.groupby(dow_col)[er_col].mean()
            vals = [means.get(d, 0) for d in range(7)]
            ax.bar(
                range(7),
                vals,
                color="#1f77b4",
                edgecolor="black",
                linewidth=0.5,
            )
            ax.set_xticks(range(7))
            ax.set_xticklabels(day_labels)

        ax.set_title("Day-of-Week Effect on Engagement")
        ax.set_ylabel("Mean Engagement Rate")
        ax.set_xlabel("Day of Week")
        if save:
            self.save_figure(fig, "day_of_week_effect.png")
        return fig

    def plot_video_age_vs_views(
        self, videos_df: pd.DataFrame, save: bool = True
    ) -> Optional[plt.Figure]:
        """Scatter plot of video age vs views with trend line."""
        age_col = _resolve_col(videos_df, ["video_age_days"])
        v_col = _resolve_col(videos_df, ["views", "view_count"])
        ch_col = self._get_channel_col(videos_df)

        if age_col is None or v_col is None:
            logger.warning("plot_video_age_vs_views: missing age or views column.")
            return None

        df = videos_df.dropna(subset=[age_col, v_col]).copy()
        if len(df) < 5:
            logger.warning("plot_video_age_vs_views: insufficient data.")
            return None

        fig, ax = plt.subplots(figsize=self.figsize)

        if ch_col and ch_col in df.columns:
            channels = sorted(df[ch_col].dropna().unique())
            palette = _get_channel_colors(channels)
            for ch in channels:
                sub = df[df[ch_col] == ch]
                ax.scatter(
                    sub[age_col],
                    sub[v_col],
                    alpha=0.4,
                    s=15,
                    label=ch,
                    color=palette.get(ch, "#1f77b4"),
                )
                # Trend line per channel
                if len(sub) > 2:
                    try:
                        coeffs = np.polyfit(sub[age_col], sub[v_col], 1)
                        poly = np.poly1d(coeffs)
                        x_range = np.linspace(
                            sub[age_col].min(), sub[age_col].max(), 100
                        )
                        ax.plot(
                            x_range,
                            poly(x_range),
                            "--",
                            linewidth=2,
                            color=palette.get(ch, "#1f77b4"),
                            alpha=0.8,
                        )
                    except Exception:
                        pass
            ax.legend(loc="best")
        else:
            ax.scatter(
                df[age_col], df[v_col], alpha=0.4, s=15, color="#1f77b4"
            )
            if len(df) > 2:
                coeffs = np.polyfit(df[age_col], df[v_col], 1)
                poly = np.poly1d(coeffs)
                x_range = np.linspace(df[age_col].min(), df[age_col].max(), 100)
                ax.plot(x_range, poly(x_range), "--", linewidth=2, color="crimson")

        ax.set_title("Video Age vs Views")
        ax.set_xlabel("Video Age (days)")
        ax.set_ylabel("Views")
        ax.set_yscale("log")
        if save:
            self.save_figure(fig, "video_age_vs_views.png")
        return fig

    def plot_comment_volume_over_time(
        self, comments_df: pd.DataFrame, save: bool = True
    ) -> Optional[plt.Figure]:
        """Line plot of comment count over time."""
        ts_col = _resolve_col(
            comments_df, ["published_at", "publishedAt", "published_date"]
        )
        if ts_col is None:
            logger.warning("plot_comment_volume_over_time: missing timestamp.")
            return None

        df = comments_df.copy()
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col]).sort_values(ts_col)
        df["_period"] = df[ts_col].dt.to_period("W").astype(str)

        vol = df.groupby("_period").size().reset_index(name="count")

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.fill_between(
            range(len(vol)),
            vol["count"],
            alpha=0.3,
            color="#1f77b4",
        )
        ax.plot(
            range(len(vol)),
            vol["count"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            color="#1f77b4",
        )
        step = max(1, len(vol) // 10)
        tick_positions = list(range(0, len(vol), step))
        tick_labels = [vol["_period"].iloc[i] for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
        ax.set_title("Comment Volume Over Time")
        ax.set_ylabel("Number of Comments")
        ax.set_xlabel("Week")
        if save:
            self.save_figure(fig, "comment_volume_over_time.png")
        return fig

    def plot_engagement_decomposition(
        self, videos_df: pd.DataFrame, save: bool = True
    ) -> Optional[plt.Figure]:
        """Stacked or grouped bar: reach, approval, discussion per channel."""
        ch_col = self._get_channel_col(videos_df)
        v_col = _resolve_col(videos_df, ["views", "view_count"])
        l_col = _resolve_col(videos_df, ["likes", "like_count"])
        c_col = _resolve_col(videos_df, ["comments", "comment_count"])

        if not all([v_col, l_col, c_col]):
            logger.warning("plot_engagement_decomposition: missing metric columns.")
            return None

        grp_col = ch_col if ch_col else None
        df = videos_df.copy()
        df["_reach"] = df[v_col]
        df["_approval"] = df[l_col]
        df["_discussion"] = df[c_col]

        if grp_col:
            agg = (
                df.groupby(grp_col)[["_reach", "_approval", "_discussion"]]
                .mean()
                .reset_index()
            )
        else:
            agg = pd.DataFrame(
                {
                    "category": ["All"],
                    "_reach": [df["_reach"].mean()],
                    "_approval": [df["_approval"].mean()],
                    "_discussion": [df["_discussion"].mean()],
                }
            )
            grp_col = "category"

        categories = sorted(agg[grp_col].unique()) if grp_col in agg.columns else agg[grp_col].tolist()
        metrics = ["_reach", "_approval", "_discussion"]
        metric_labels = ["Reach (Views)", "Approval (Likes)", "Discussion (Comments)"]

        x = np.arange(len(categories))
        width = 0.25
        fig, ax = plt.subplots(figsize=self.figsize)

        for i, (mcol, mlabel) in enumerate(zip(metrics, metric_labels)):
            vals = agg[mcol].values if mcol in agg.columns else np.zeros(len(categories))
            ax.bar(
                x + i * width - width,
                vals,
                width,
                label=mlabel,
                edgecolor="black",
                linewidth=0.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=15, ha="right")
        ax.set_title("Engagement Decomposition by Channel")
        ax.set_ylabel("Mean Count")
        ax.legend(loc="best")
        ax.set_yscale("log")
        if save:
            self.save_figure(fig, "engagement_decomposition.png")
        return fig

    def plot_rolling_average(
        self,
        time_series: pd.DataFrame,
        window: int = 7,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Rolling average of engagement metrics over time.

        Expects time_series with a DatetimeIndex or a 'published_at' column.
        """
        if isinstance(time_series.index, pd.DatetimeIndex):
            df = time_series.sort_index().copy()
            date_index = True
        else:
            ts_col = _resolve_col(time_series, ["published_at", "publishedAt"])
            if ts_col is None:
                # Try index
                if isinstance(time_series.index, pd.DatetimeIndex):
                    df = time_series.sort_index().copy()
                    date_index = True
                else:
                    logger.warning("plot_rolling_average: no timestamp found.")
                    return None
            else:
                df = time_series.copy()
                df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
                df = df.dropna(subset=[ts_col]).set_index(ts_col).sort_index()
                date_index = True

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        metric_cols = [
            c
            for c in numeric_cols
            if c.lower()
            in {
                "views",
                "view_count",
                "likes",
                "like_count",
                "comments",
                "comment_count",
                "engagement_rate",
            }
        ]
        if not metric_cols:
            metric_cols = numeric_cols[:5]

        fig, ax = plt.subplots(figsize=self.figsize)

        for col in metric_cols[:5]:
            series = df[col].dropna()
            if len(series) > window:
                rolled = series.rolling(window=window).mean()
                ax.plot(
                    range(len(rolled)),
                    rolled,
                    linewidth=1.5,
                    label=col.replace("_", " ").title(),
                )

        step = max(1, len(df) // 8)
        tick_positions = list(range(0, len(df), step))
        tick_labels = [
            str(df.index[i].strftime("%Y-%m-%d")) if hasattr(df.index[i], "strftime") else str(df.index[i])
            for i in tick_positions
        ]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

        ax.set_title(f"Rolling Average (window={window})")
        ax.set_ylabel("Value")
        ax.legend(loc="best")
        if save:
            self.save_figure(fig, "rolling_average.png")
        return fig

    # ==================================================================
    # 3. SENTIMENT & NLP VISUALIZATIONS
    # ==================================================================

    def plot_sentiment_distribution(
        self, comments_df: pd.DataFrame, save: bool = True
    ) -> Optional[plt.Figure]:
        """Pie chart of sentiment labels."""
        label_col = _resolve_col(comments_df, ["sentiment_label"])
        if label_col is None:
            logger.warning("plot_sentiment_distribution: missing sentiment_label.")
            return None

        counts = comments_df[label_col].value_counts()
        colors_map = {
            "positive": "#2ca02c",
            "negative": "#d62728",
            "neutral": "#7f7f7f",
        }
        colors_list = [colors_map.get(str(k).lower(), "#1f77b4") for k in counts.index]

        fig, ax = plt.subplots(figsize=(7, 7))
        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors_list,
            wedgeprops={"edgecolor": "white", "linewidth": 1},
        )
        for t in autotexts:
            t.set_fontsize(10)
        ax.set_title("Sentiment Distribution")
        if save:
            self.save_figure(fig, "sentiment_distribution.png")
        return fig

    def plot_sentiment_by_channel(
        self,
        videos_df: pd.DataFrame,
        comments_df: pd.DataFrame,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Grouped bar chart of sentiment proportions by channel."""
        sent_col = _resolve_col(comments_df, ["sentiment_label"])
        vid_col = _resolve_col(comments_df, ["video_id"])
        ch_col = _resolve_col(videos_df, ["channel_name", "channel_title", "channel"])
        vid_ch_col = _resolve_col(videos_df, ["video_id"])

        if not all([sent_col, vid_col]):
            logger.warning("plot_sentiment_by_channel: missing sentiment/video columns.")
            return None

        merged = comments_df[[vid_col, sent_col]].merge(
            videos_df[[vid_ch_col, ch_col]].drop_duplicates(subset=[vid_ch_col])
            if vid_ch_col and ch_col
            else pd.DataFrame(columns=[vid_col, "channel"]),
            left_on=vid_col,
            right_on=vid_ch_col or vid_col,
            how="left",
        )

        if ch_col is None or ch_col not in merged.columns:
            merged["channel"] = "Unknown"

        cross = (
            merged.groupby(["channel", sent_col]).size().unstack(fill_value=0)
        )
        props = cross.div(cross.sum(axis=1), axis=0)

        labels_order = ["positive", "neutral", "negative"]
        existing_labels = [l for l in labels_order if l in props.columns]
        props = props[existing_labels]

        channels = props.index.tolist()
        palette = _get_channel_colors(channels)

        fig, ax = plt.subplots(figsize=self.figsize)
        x = np.arange(len(channels))
        width = 0.25
        sent_colors = {
            "positive": "#2ca02c",
            "neutral": "#7f7f7f",
            "negative": "#d62728",
        }

        for i, label in enumerate(existing_labels):
            vals = props[label].values
            ax.bar(
                x + i * width,
                vals,
                width,
                label=label.title(),
                color=sent_colors.get(label, "#1f77b4"),
                edgecolor="black",
                linewidth=0.5,
            )

        ax.set_xticks(x + width)
        ax.set_xticklabels(channels, rotation=15, ha="right")
        ax.set_title("Sentiment Distribution by Channel")
        ax.set_ylabel("Proportion")
        ax.legend(loc="best")
        ax.set_ylim(0, 1)
        if save:
            self.save_figure(fig, "sentiment_by_channel.png")
        return fig

    def plot_sentiment_timeline(
        self,
        comments_df: pd.DataFrame,
        time_column: str = "published_at",
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Sentiment trend over time (line per sentiment label)."""
        ts_col = _resolve_col(comments_df, [time_column, "publishedAt"])
        sent_col = _resolve_col(comments_df, ["sentiment_label"])

        if ts_col is None or sent_col is None:
            logger.warning("plot_sentiment_timeline: missing time or sentiment column.")
            return None

        df = comments_df.copy()
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col]).sort_values(ts_col)
        df["_period"] = df[ts_col].dt.to_period("W").astype(str)

        pivot = (
            df.groupby(["_period", sent_col])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )
        props = pivot.div(pivot.sum(axis=1), axis=0)

        fig, ax = plt.subplots(figsize=self.figsize)
        sent_colors = {
            "positive": "#2ca02c",
            "neutral": "#7f7f7f",
            "negative": "#d62728",
        }

        for label in props.columns:
            ax.plot(
                range(len(props)),
                props[label],
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=label.title(),
                color=sent_colors.get(str(label).lower(), "#1f77b4"),
            )

        step = max(1, len(props) // 10)
        tick_positions = list(range(0, len(props), step))
        tick_labels = [props.index[i] for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
        ax.set_title("Sentiment Timeline")
        ax.set_ylabel("Proportion")
        ax.legend(loc="best")
        ax.set_ylim(0, 1)
        if save:
            self.save_figure(fig, "sentiment_timeline.png")
        return fig

    def plot_topic_distribution(
        self,
        comments_df: pd.DataFrame,
        topic_column: str = "topic_label",
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Horizontal bar chart of topic frequencies."""
        topic_col = _resolve_col(comments_df, [topic_column, "topic_id"])
        if topic_col is None:
            logger.warning("plot_topic_distribution: missing topic column.")
            return None

        counts = comments_df[topic_col].value_counts().head(20)

        fig, ax = plt.subplots(figsize=(10, max(6, len(counts) * 0.35)))
        ax.barh(
            counts.index[::-1],
            counts.values[::-1],
            color="#1f77b4",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_title("Topic Distribution")
        ax.set_xlabel("Number of Comments")
        if save:
            self.save_figure(fig, "topic_distribution.png")
        return fig

    def plot_topic_trends(
        self,
        topic_timeseries: pd.DataFrame,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Line plot of topic frequencies over time."""
        if topic_timeseries.empty:
            logger.warning("plot_topic_trends: empty timeseries.")
            return None

        top_topics = topic_timeseries.sum().nlargest(8).index.tolist()

        fig, ax = plt.subplots(figsize=self.figsize)
        for i, topic in enumerate(top_topics):
            ax.plot(
                range(len(topic_timeseries)),
                topic_timeseries[topic].values,
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=str(topic)[:40],
                color=CHANNEL_COLORS_FALLBACK[
                    i % len(CHANNEL_COLORS_FALLBACK)
                ],
            )

        step = max(1, len(topic_timeseries) // 8)
        tick_positions = list(range(0, len(topic_timeseries), step))
        tick_labels = [str(idx) for idx in topic_timeseries.index[::step]]
        # Pad if needed
        tick_labels_full = [str(idx) for idx in topic_timeseries.index]
        actual_labels = [
            tick_labels_full[i] if i < len(tick_labels_full) else ""
            for i in tick_positions
        ]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(actual_labels, rotation=45, ha="right", fontsize=8)

        ax.set_title("Topic Trends Over Time")
        ax.set_ylabel("Comment Count")
        ax.legend(loc="best", fontsize=8, ncol=2)
        if save:
            self.save_figure(fig, "topic_trends.png")
        return fig

    def plot_embedding_projection(
        self,
        embeddings: np.ndarray,
        labels: Union[np.ndarray, List[str]],
        method: str = "pca",
        title: str = "Embedding Projection",
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """2D scatter of embeddings coloured by labels (PCA or UMAP)."""
        if len(embeddings) < 2:
            logger.warning("plot_embedding_projection: need at least 2 embeddings.")
            return None

        if method == "umap":
            try:
                import umap

                reducer = umap.UMAP(
                    n_components=2,
                    random_state=CONFIG["RANDOM_STATE"],
                    n_neighbors=min(15, len(embeddings) - 1),
                    min_dist=0.1,
                )
                coords = reducer.fit_transform(embeddings)
            except ImportError:
                logger.warning("umap not available; falling back to PCA.")
                method = "pca"

        if method == "pca":
            from sklearn.decomposition import PCA

            pca = PCA(n_components=2, random_state=CONFIG["RANDOM_STATE"])
            coords = pca.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=self.figsize)

        unique_labels = sorted(set(str(l) for l in labels))
        palette = (
            _get_channel_colors(unique_labels)
            if all(l in CHANNEL_COLORS for l in unique_labels)
            else {}
        )

        for i, lbl in enumerate(unique_labels):
            mask = np.array([str(l) == lbl for l in labels])
            color = palette.get(lbl, CHANNEL_COLORS_FALLBACK[i % len(CHANNEL_COLORS_FALLBACK)])
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                alpha=0.5,
                s=10,
                label=lbl,
                color=color,
            )

        ax.set_title(title)
        ax.set_xlabel(f"{method.upper()} Component 1")
        ax.set_ylabel(f"{method.upper()} Component 2")
        if len(unique_labels) <= 10:
            ax.legend(loc="best", fontsize=8, markerscale=2)
        if save:
            self.save_figure(fig, f"embedding_projection_{method}.png")
        return fig

    # ==================================================================
    # 4. MODEL VISUALIZATIONS
    # ==================================================================

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Horizontal bar chart of feature importance."""
        if importance_df.empty:
            logger.warning("plot_feature_importance: empty DataFrame.")
            return None

        # Auto-detect columns
        cols = importance_df.columns.tolist()
        feat_col = (
            "feature" if "feature" in cols else cols[0]
        )
        imp_col = (
            "importance" if "importance" in cols else cols[1] if len(cols) > 1 else None
        )
        if imp_col is None:
            logger.warning("plot_feature_importance: could not detect importance column.")
            return None

        df = importance_df.nlargest(top_n, imp_col).copy()
        df = df.sort_values(imp_col)

        fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.35)))
        ax.barh(
            df[feat_col],
            df[imp_col],
            color="#1f77b4",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_title(f"Top {len(df)} Feature Importance")
        ax.set_xlabel("Importance")
        if save:
            self.save_figure(fig, "feature_importance.png")
        return fig

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Seaborn heatmap of confusion matrix."""
        fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.8), max(6, len(class_names) * 0.8)))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            square=True,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title("Confusion Matrix")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        if save:
            self.save_figure(fig, "confusion_matrix.png")
        return fig

    def plot_roc_curves(
        self,
        fpr_dict: Dict[str, np.ndarray],
        tpr_dict: Dict[str, np.ndarray],
        roc_auc_dict: Dict[str, float],
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """ROC curves for multi-class (one-vs-rest)."""
        fig, ax = plt.subplots(figsize=(8, 8))

        for i, (cls, fpr) in enumerate(fpr_dict.items()):
            tpr = tpr_dict.get(cls)
            auc_val = roc_auc_dict.get(cls, 0.0)
            if tpr is not None:
                ax.plot(
                    fpr,
                    tpr,
                    linewidth=2,
                    label=f"{cls} (AUC={auc_val:.3f})",
                    color=CHANNEL_COLORS_FALLBACK[
                        i % len(CHANNEL_COLORS_FALLBACK)
                    ],
                )

        ax.plot(
            [0, 1],
            [0, 1],
            "k--",
            linewidth=1,
            alpha=0.5,
            label="Random (AUC=0.500)",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves (One-vs-Rest)")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        if save:
            self.save_figure(fig, "roc_curves.png")
        return fig

    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Residual plot: scatter + histogram."""
        residuals = np.array(y_true) - np.array(y_pred)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].scatter(
            y_pred,
            residuals,
            alpha=0.4,
            s=10,
            color="#1f77b4",
            edgecolors="none",
        )
        axes[0].axhline(0, color="crimson", linestyle="--", linewidth=1.5)
        axes[0].set_xlabel("Predicted Values")
        axes[0].set_ylabel("Residuals (Actual - Predicted)")
        axes[0].set_title("Residual Plot")
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(
            residuals,
            bins=40,
            color="#2ca02c",
            edgecolor="white",
            alpha=0.7,
        )
        axes[1].axvline(0, color="crimson", linestyle="--", linewidth=1.5)
        axes[1].set_xlabel("Residuals")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Residual Distribution")

        fig.suptitle("Model Residual Analysis", fontsize=14)
        if save:
            self.save_figure(fig, "residuals.png")
        return fig

    def plot_predictions_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Scatter of predictions vs actual with identity line."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        fig, ax = plt.subplots(figsize=(7, 7))
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())

        ax.scatter(
            y_true,
            y_pred,
            alpha=0.4,
            s=15,
            color="#1f77b4",
            edgecolors="none",
        )
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "k--",
            linewidth=1.5,
            alpha=0.7,
            label="Identity Line",
        )

        # R² annotation if possible
        try:
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            ax.text(
                0.05,
                0.95,
                f"$R^2$ = {r2:.3f}",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
        except Exception:
            pass

        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Predictions vs Actual")
        ax.legend(loc="best")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        if save:
            self.save_figure(fig, "predictions_vs_actual.png")
        return fig

    def plot_model_comparison(
        self,
        metrics_dict: Dict[str, float],
        metric: str = "R2",
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Bar chart comparing models on a given metric."""
        if not metrics_dict:
            logger.warning("plot_model_comparison: empty metrics dict.")
            return None

        models = list(metrics_dict.keys())
        values = list(metrics_dict.values())

        fig, ax = plt.subplots(figsize=(max(6, len(models) * 0.8), 5))
        bars = ax.bar(
            models,
            values,
            color=CHANNEL_COLORS_FALLBACK[: len(models)],
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:.4f}",
                ha="center",
                fontsize=9,
            )

        ax.set_title(f"Model Comparison ({metric})")
        ax.set_ylabel(metric)
        ax.set_xticklabels(models, rotation=15, ha="right")
        if save:
            self.save_figure(fig, "model_comparison.png")
        return fig

    # ==================================================================
    # 5. CLUSTERING VISUALIZATIONS
    # ==================================================================

    def plot_cluster_scatter(
        self,
        features_2d: np.ndarray,
        labels: np.ndarray,
        title: str = "Video Clusters",
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """2D scatter coloured by cluster assignment."""
        if len(features_2d) < 2:
            logger.warning("plot_cluster_scatter: insufficient data.")
            return None

        unique_clusters = sorted(set(labels))
        n_clusters = len(unique_clusters)

        fig, ax = plt.subplots(figsize=self.figsize)
        cmap = plt.cm.get_cmap("tab10", max(n_clusters, 3))

        for i, cluster in enumerate(unique_clusters):
            mask = labels == cluster
            lbl = f"Cluster {cluster}" if cluster >= 0 else "Noise"
            ax.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                alpha=0.6,
                s=15,
                label=lbl,
                color=cmap(i),
                edgecolors="none",
            )

        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        if n_clusters <= 12:
            ax.legend(loc="best", fontsize=8, markerscale=2)
        if save:
            self.save_figure(fig, "cluster_scatter.png")
        return fig

    def plot_cluster_profiles(
        self,
        cluster_profiles: pd.DataFrame,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Radar chart for cluster comparison, or grouped bar fallback."""
        if cluster_profiles.empty:
            logger.warning("plot_cluster_profiles: empty DataFrame.")
            return None

        # Attempt radar chart
        try:
            fig = self._plot_radar(cluster_profiles)
            if save:
                self.save_figure(fig, "cluster_profiles_radar.png")
            return fig
        except Exception:
            pass

        # Fallback: grouped bar chart
        clusters = cluster_profiles.index.astype(str).tolist()
        metrics_cols = cluster_profiles.select_dtypes(include=[np.number]).columns.tolist()
        if not metrics_cols:
            logger.warning("plot_cluster_profiles: no numeric columns.")
            return None

        fig, ax = plt.subplots(figsize=(max(8, len(metrics_cols) * 1.2), 6))
        x = np.arange(len(metrics_cols))
        width = 0.8 / max(1, len(clusters))

        for i, cluster in enumerate(clusters):
            vals = cluster_profiles.loc[
                cluster_profiles.index.isin([cluster]) or cluster_profiles.index.astype(str) == cluster,
                metrics_cols,
            ].values
            if len(vals) > 0:
                ax.bar(
                    x + i * width,
                    vals[0],
                    width,
                    label=f"Cluster {cluster}",
                    color=CHANNEL_COLORS_FALLBACK[i % len(CHANNEL_COLORS_FALLBACK)],
                    edgecolor="black",
                    linewidth=0.5,
                )

        ax.set_xticks(x + width * (len(clusters) - 1) / 2)
        ax.set_xticklabels(metrics_cols, rotation=25, ha="right")
        ax.set_title("Cluster Profiles")
        ax.legend(loc="best", fontsize=8)
        if save:
            self.save_figure(fig, "cluster_profiles.png")
        return fig

    def _plot_radar(self, cluster_profiles: pd.DataFrame) -> plt.Figure:
        """Internal radar/spider chart for cluster profiles."""
        metrics_cols = cluster_profiles.select_dtypes(include=[np.number]).columns.tolist()
        num_vars = len(metrics_cols)
        if num_vars < 3:
            raise ValueError("Need at least 3 metrics for radar chart.")

        # Normalise to [0,1]
        profile_norm = cluster_profiles[metrics_cols].copy()
        for col in metrics_cols:
            mx = profile_norm[col].max()
            mn = profile_norm[col].min()
            if mx > mn:
                profile_norm[col] = (profile_norm[col] - mn) / (mx - mn)
            else:
                profile_norm[col] = 0.5

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

        for i, (idx, row) in enumerate(profile_norm.iterrows()):
            values = row.values.tolist()
            values += values[:1]
            ax.plot(
                angles,
                values,
                "o-",
                linewidth=2,
                label=f"Cluster {idx}",
                color=CHANNEL_COLORS_FALLBACK[i % len(CHANNEL_COLORS_FALLBACK)],
            )
            ax.fill(
                angles,
                values,
                alpha=0.1,
                color=CHANNEL_COLORS_FALLBACK[i % len(CHANNEL_COLORS_FALLBACK)],
            )

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_cols, fontsize=9)
        ax.set_title("Cluster Profiles (Radar)", y=1.08, fontsize=14)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
        return fig

    def plot_silhouette(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Silhouette plot for cluster quality."""
        try:
            from sklearn.metrics import silhouette_samples, silhouette_score
        except ImportError:
            logger.warning("plot_silhouette: scikit-learn required.")
            return None

        unique_labels = sorted(set(labels) - {-1})
        if len(unique_labels) < 2:
            logger.warning("plot_silhouette: need at least 2 clusters.")
            return None

        n_clusters = len(unique_labels)
        silhouette_avg = silhouette_score(features, labels)

        sample_silhouette_values = silhouette_samples(features, labels)

        fig, ax = plt.subplots(figsize=self.figsize)
        y_lower = 10
        cmap = plt.cm.get_cmap("tab10", n_clusters)

        for i, cluster in enumerate(unique_labels):
            cluster_values = sample_silhouette_values[labels == cluster]
            cluster_values.sort()
            size_cluster = len(cluster_values)
            y_upper = y_lower + size_cluster

            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_values,
                facecolor=cmap(i),
                edgecolor=cmap(i),
                alpha=0.6,
                label=f"Cluster {cluster}",
            )
            y_lower = y_upper + 10

        ax.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2, label=f"Average ({silhouette_avg:.3f})")
        ax.set_title("Silhouette Plot")
        ax.set_xlabel("Silhouette Coefficient")
        ax.set_ylabel("Cluster")
        ax.set_yticks([])
        ax.legend(loc="best", fontsize=8)
        if save:
            self.save_figure(fig, "silhouette.png")
        return fig

    # ==================================================================
    # 6. GRAPH VISUALIZATIONS
    # ==================================================================

    def plot_graph_summary(
        self, graph_stats: Dict[str, Any], save: bool = True
    ) -> Optional[plt.Figure]:
        """Bar chart of graph statistics."""
        numeric_stats = {
            k: v for k, v in graph_stats.items() if isinstance(v, (int, float))
        }
        if not numeric_stats:
            logger.warning("plot_graph_summary: no numeric graph stats.")
            return None

        fig, ax = plt.subplots(figsize=(max(6, len(numeric_stats) * 1), 5))
        labels = list(numeric_stats.keys())
        values = list(numeric_stats.values())
        ax.bar(
            labels,
            values,
            color=CHANNEL_COLORS_FALLBACK[: len(labels)],
            edgecolor="black",
            linewidth=0.5,
        )
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right")
        for bar, val in zip(ax.containers[0], values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                f"{val:.4g}" if isinstance(val, float) else str(val),
                ha="center",
                fontsize=8,
            )
        ax.set_title("Graph Summary Statistics")
        if save:
            self.save_figure(fig, "graph_summary.png")
        return fig

    def plot_centrality_distribution(
        self,
        central_nodes: pd.DataFrame,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Bar chart of top central nodes by centrality score."""
        if central_nodes.empty:
            logger.warning("plot_centrality_distribution: empty DataFrame.")
            return None

        # Auto-detect columns
        cols = central_nodes.columns.tolist()
        node_col = next((c for c in cols if c in {"node", "nodes", "channel", "video_id", "name"}), cols[0])
        cent_col = next(
            (c for c in cols if c in {"centrality", "score", "value", "degree"}),
            cols[1] if len(cols) > 1 else None,
        )
        if cent_col is None:
            logger.warning("plot_centrality_distribution: could not detect centrality column.")
            return None

        df = central_nodes.sort_values(cent_col, ascending=False).head(30).copy()
        df = df.sort_values(cent_col)

        fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.35)))
        ax.barh(
            df[node_col].astype(str).str[:40],
            df[cent_col],
            color="#1f77b4",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_title("Top Nodes by Centrality")
        ax.set_xlabel("Centrality Score")
        if save:
            self.save_figure(fig, "centrality_distribution.png")
        return fig

    def plot_community_sizes(
        self,
        communities: Dict[Any, List[Any]],
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Bar chart of community sizes."""
        if not communities:
            logger.warning("plot_community_sizes: empty communities dict.")
            return None

        sizes = {str(k): len(v) for k, v in communities.items()}
        sorted_sizes = dict(sorted(sizes.items(), key=lambda x: x[1], reverse=True))

        fig, ax = plt.subplots(figsize=(max(6, len(sorted_sizes) * 0.4), 5))
        ax.bar(
            sorted_sizes.keys(),
            sorted_sizes.values(),
            color=CHANNEL_COLORS_FALLBACK[: len(sorted_sizes)],
            edgecolor="black",
            linewidth=0.5,
        )
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right")
        ax.set_title("Community Sizes")
        ax.set_ylabel("Number of Nodes")
        if save:
            self.save_figure(fig, "community_sizes.png")
        return fig

    def plot_community_composition(
        self,
        community_stats: pd.DataFrame,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Stacked bar chart of node types per community."""
        if community_stats.empty:
            logger.warning("plot_community_composition: empty DataFrame.")
            return None

        # Assume first column is community id, rest are type counts
        if community_stats.shape[1] < 2:
            logger.warning("plot_community_composition: need at least 2 columns.")
            return None

        id_col = community_stats.columns[0]
        type_cols = community_stats.columns[1:].tolist()

        fig, ax = plt.subplots(figsize=self.figsize)
        bottom = np.zeros(len(community_stats))
        for i, col in enumerate(type_cols):
            ax.bar(
                community_stats[id_col].astype(str),
                community_stats[col],
                bottom=bottom,
                label=col,
                color=CHANNEL_COLORS_FALLBACK[i % len(CHANNEL_COLORS_FALLBACK)],
                edgecolor="black",
                linewidth=0.3,
            )
            bottom += community_stats[col].values

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right")
        ax.set_title("Community Composition by Node Type")
        ax.set_ylabel("Count")
        ax.legend(loc="best", fontsize=8)
        if save:
            self.save_figure(fig, "community_composition.png")
        return fig

    # ==================================================================
    # 7. INTERACTIVE PLOTLY VISUALIZATIONS
    # ==================================================================

    def interactive_engagement_scatter(
        self, videos_df: pd.DataFrame, save: bool = True
    ) -> Optional[Any]:
        """Plotly scatter: views vs likes, coloured by channel, hover shows title."""
        try:
            import plotly.express as px
        except ImportError:
            logger.warning("plotly not available; skipping interactive_engagement_scatter.")
            return None

        cols = _resolve_many(
            videos_df,
            {
                "views": ["views", "view_count"],
                "likes": ["likes", "like_count"],
                "channel": ["channel_name", "channel_title", "channel"],
                "title": ["title"],
            },
        )
        if "views" not in cols or "likes" not in cols:
            logger.warning("interactive_engagement_scatter: missing views/likes.")
            return None

        hover_data = [cols["title"]] if "title" in cols else None
        fig = px.scatter(
            videos_df,
            x=cols["views"],
            y=cols["likes"],
            color=cols.get("channel"),
            hover_data=hover_data,
            title="Views vs Likes by Channel",
            log_x=True,
            log_y=True,
            opacity=0.6,
            color_discrete_sequence=list(CHANNEL_COLORS.values()),
        )
        fig.update_layout(template="plotly_white")
        if save:
            path = os.path.join(CONFIG["FIGURES_DIR"], "interactive_engagement_scatter.html")
            fig.write_html(path)
            logger.info("Saved interactive plot → %s", path)
        return fig

    def interactive_timeline(
        self, videos_df: pd.DataFrame, save: bool = True
    ) -> Optional[Any]:
        """Plotly line chart with range slider: engagement over time."""
        try:
            import plotly.express as px
        except ImportError:
            logger.warning("plotly not available; skipping interactive_timeline.")
            return None

        ts_col = _resolve_col(videos_df, ["published_at", "publishedAt"])
        er_col = _resolve_col(videos_df, ["engagement_rate"])
        ch_col = self._get_channel_col(videos_df)

        if ts_col is None:
            logger.warning("interactive_timeline: missing timestamp.")
            return None

        df = videos_df.copy()
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col])

        if er_col is None:
            v_col = _resolve_col(df, ["views", "view_count"])
            l_col = _resolve_col(df, ["likes", "like_count"])
            c_col = _resolve_col(df, ["comments", "comment_count"])
            if v_col and l_col and c_col:
                df["engagement_rate"] = np.where(
                    df[v_col] > 0,
                    (df[l_col] + df[c_col]) / df[v_col],
                    0.0,
                )
                er_col = "engagement_rate"

        if er_col is None:
            logger.warning("interactive_timeline: missing engagement rate.")
            return None

        label_col = ch_col if ch_col else None
        if label_col is None:
            label_col = "All"
            df["All"] = "All Videos"

        fig = px.line(
            df.sort_values(ts_col),
            x=ts_col,
            y=er_col,
            color=label_col,
            title="Engagement Rate Over Time",
            color_discrete_sequence=list(CHANNEL_COLORS.values()),
        )
        fig.update_xaxes(rangeslider_visible=True)
        fig.update_layout(template="plotly_white")
        if save:
            path = os.path.join(CONFIG["FIGURES_DIR"], "interactive_timeline.html")
            fig.write_html(path)
            logger.info("Saved interactive plot → %s", path)
        return fig

    def interactive_topic_explorer(
        self, topic_data: pd.DataFrame, save: bool = True
    ) -> Optional[Any]:
        """Plotly sunburst or treemap of topics."""
        try:
            import plotly.express as px
        except ImportError:
            logger.warning("plotly not available; skipping interactive_topic_explorer.")
            return None

        if topic_data.empty:
            logger.warning("interactive_topic_explorer: empty DataFrame.")
            return None

        cols = topic_data.columns.tolist()
        label_col = next(
            (c for c in cols if c in {"topic_label", "topic", "label"}), cols[0]
        )
        value_col = next(
            (c for c in cols if c in {"count", "size", "value", "proportion"}),
            None,
        )

        if value_col is None:
            # Create a simple treemap from value counts
            topic_counts = topic_data[label_col].value_counts().reset_index()
            topic_counts.columns = ["topic", "count"]
            label_col = "topic"
            value_col = "count"
            plot_df = topic_counts
        else:
            plot_df = topic_data

        fig = px.treemap(
            plot_df,
            path=[label_col],
            values=value_col,
            title="Topic Explorer",
            color=value_col,
            color_continuous_scale="Blues",
        )
        fig.update_layout(template="plotly_white")
        if save:
            path = os.path.join(CONFIG["FIGURES_DIR"], "interactive_topic_explorer.html")
            fig.write_html(path)
            logger.info("Saved interactive plot → %s", path)
        return fig

    def interactive_channel_dashboard(
        self,
        videos_df: pd.DataFrame,
        comments_df: pd.DataFrame,
        save: bool = True,
    ) -> Optional[Any]:
        """Plotly subplots dashboard: views dist, sentiment, engagement, top videos."""
        try:
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            logger.warning("plotly not available; skipping interactive_channel_dashboard.")
            return None

        ch_col = self._get_channel_col(videos_df)
        if ch_col is None:
            logger.warning("interactive_channel_dashboard: missing channel column.")
            return None

        channels = sorted(videos_df[ch_col].dropna().unique())
        n_channels = len(channels)
        if n_channels == 0:
            return None

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Views Distribution",
                "Sentiment Distribution",
                "Engagement Timeline",
                "Top 10 Videos by Views",
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
        )

        palette = _get_channel_colors(channels)
        v_col = _resolve_col(videos_df, ["views", "view_count"]) or "views"
        er_col = _resolve_col(videos_df, ["engagement_rate"])
        ts_col = _resolve_col(videos_df, ["published_at", "publishedAt"])
        sent_col = _resolve_col(comments_df, ["sentiment_label"])

        # 1. Views distribution
        for ch in channels:
            data = videos_df[videos_df[ch_col] == ch][v_col].dropna()
            data_log = np.log1p(data[data > 0])
            fig.add_trace(
                go.Histogram(
                    x=data_log,
                    name=ch,
                    opacity=0.6,
                    marker_color=palette.get(ch),
                    legendgroup=ch,
                ),
                row=1,
                col=1,
            )

        # 2. Sentiment distribution
        if sent_col:
            vid_col = _resolve_col(comments_df, ["video_id"])
            vid_ch_col = _resolve_col(videos_df, ["video_id"])
            if vid_col and vid_ch_col:
                merged = comments_df[[vid_col, sent_col]].merge(
                    videos_df[[vid_ch_col, ch_col]],
                    left_on=vid_col,
                    right_on=vid_ch_col,
                    how="left",
                )
                sent_order = ["positive", "neutral", "negative"]
                for i, sentiment in enumerate(sent_order):
                    counts = []
                    for ch in channels:
                        sub = merged[merged[ch_col] == ch]
                        cnt = (sub[sent_col] == sentiment).sum()
                        counts.append(cnt)
                    fig.add_trace(
                        go.Bar(
                            x=channels,
                            y=counts,
                            name=sentiment.title(),
                            marker_color={
                                "positive": "#2ca02c",
                                "neutral": "#7f7f7f",
                                "negative": "#d62728",
                            }.get(sentiment, "#1f77b4"),
                            legendgroup=f"sent_{sentiment}",
                        ),
                        row=1,
                        col=2,
                    )

        # 3. Engagement timeline
        if ts_col and er_col:
            for ch in channels:
                sub = videos_df[videos_df[ch_col] == ch].copy()
                sub[ts_col] = pd.to_datetime(sub[ts_col], errors="coerce")
                sub = sub.dropna(subset=[ts_col]).sort_values(ts_col)
                fig.add_trace(
                    go.Scatter(
                        x=sub[ts_col],
                        y=sub[er_col],
                        mode="lines+markers",
                        name=ch,
                        marker_color=palette.get(ch),
                        legendgroup=ch,
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )

        # 4. Top 10 videos
        top_videos = videos_df.nlargest(10, v_col)
        title_col = _resolve_col(videos_df, ["title"])
        for i, ch in enumerate(channels):
            sub = top_videos[top_videos[ch_col] == ch]
            if len(sub) > 0:
                fig.add_trace(
                    go.Bar(
                        y=sub[title_col].str[:50] if title_col else sub.index.astype(str),
                        x=sub[v_col],
                        name=ch,
                        orientation="h",
                        marker_color=palette.get(ch),
                        legendgroup=ch,
                        showlegend=False,
                    ),
                    row=2,
                    col=2,
                )

        fig.update_layout(
            height=900,
            title_text="Channel Performance Dashboard",
            template="plotly_white",
            barmode="overlay",
        )
        fig.update_xaxes(title_text="log(Views + 1)", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Channel", row=1, col=2)
        fig.update_yaxes(title_text="Comment Count", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Engagement Rate", row=2, col=1)
        fig.update_xaxes(title_text="Views", row=2, col=2)

        if save:
            path = os.path.join(CONFIG["FIGURES_DIR"], "channel_dashboard.html")
            fig.write_html(path)
            logger.info("Saved interactive dashboard → %s", path)
        return fig

    # ==================================================================
    # 8. DASHBOARD
    # ==================================================================

    def create_streamlit_dashboard_code(self) -> str:
        """Generate a standalone Streamlit dashboard script and save to outputs/."""
        script = textwrap.dedent(f'''\
            """
            Digital Media Analytics — Streamlit Dashboard
            Auto-generated by VisualizationEngine
            """
            import os
            import sys

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            import streamlit as st

            st.set_page_config(
                page_title="Digital Media Analytics",
                layout="wide",
                initial_sidebar_state="expanded",
            )

            # ------------------------------------------------------------------
            # Config
            # ------------------------------------------------------------------
            DATA_DIR = "{CONFIG['PROCESSED_DATA_DIR']}"
            FIGURES_DIR = "{CONFIG['FIGURES_DIR']}"
            CHANNEL_COLORS = {json.dumps(CHANNEL_COLORS)}

            sns.set_style("whitegrid")

            # ------------------------------------------------------------------
            # Data loading
            # ------------------------------------------------------------------
            @st.cache_data
            def load_data():
                """Load processed parquet files."""
                dfs = {{}}
                for name in ["videos_processed", "comments_processed", "channels_processed"]:
                    path = os.path.join(DATA_DIR, f"{{name}}.parquet")
                    try:
                        dfs[name] = pd.read_parquet(path)
                    except FileNotFoundError:
                        st.warning(f"File not found: {{path}}")
                        dfs[name] = pd.DataFrame()
                return dfs

            data = load_data()
            videos_df = data.get("videos_processed", pd.DataFrame())
            comments_df = data.get("comments_processed", pd.DataFrame())

            # ------------------------------------------------------------------
            # Sidebar
            # ------------------------------------------------------------------
            st.sidebar.title("Navigation")
            section = st.sidebar.radio(
                "Go to",
                [
                    "Channel Overview",
                    "Video Performance",
                    "Audience Sentiment",
                    "Topic Trends",
                    "Graph Influence Map",
                    "Strategic Recommendations",
                ],
            )

            # ------------------------------------------------------------------
            # Sections
            # ------------------------------------------------------------------
            if section == "Channel Overview":
                st.title("Channel Overview")
                if videos_df.empty:
                    st.info("No video data loaded. Run data collection and preprocessing first.")
                else:
                    ch_col = next(
                        (c for c in videos_df.columns if c.lower() in {{"channel_name", "channel_title", "channel"}}),
                        None,
                    )
                    if ch_col:
                        channels = videos_df[ch_col].unique()
                        cols = st.columns(len(channels))
                        for i, ch in enumerate(channels):
                            sub = videos_df[videos_df[ch_col] == ch]
                            with cols[i]:
                                color = CHANNEL_COLORS.get(ch, "#1f77b4")
                                st.markdown(
                                    f'<h3 style="color:{{color}}">{{ch}}</h3>',
                                    unsafe_allow_html=True,
                                )
                                st.metric("Videos", len(sub))
                                for metric in ["views", "likes", "comments"]:
                                    col = next(
                                        (c for c in sub.columns if c.lower() == metric),
                                        None,
                                    )
                                    if col:
                                        st.metric(
                                            f"Avg {{metric.title()}}",
                                            f"{{sub[col].mean():,.0f}}",
                                        )
                    st.subheader("Views Distribution")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    v_col = next(
                        (c for c in videos_df.columns if c.lower() in {{"views", "view_count"}}),
                        None,
                    )
                    if v_col and ch_col:
                        for ch in channels:
                            data_ch = videos_df[videos_df[ch_col] == ch][v_col].dropna()
                            sns.histplot(
                                np.log1p(data_ch),
                                label=ch,
                                kde=True,
                                alpha=0.4,
                                color=CHANNEL_COLORS.get(ch),
                                ax=ax,
                            )
                        ax.legend()
                        ax.set_xlabel("log(Views + 1)")
                        st.pyplot(fig)

            elif section == "Video Performance":
                st.title("Video Performance")
                if videos_df.empty:
                    st.info("No video data loaded.")
                else:
                    v_col = next(
                        (c for c in videos_df.columns if c.lower() in {{"views", "view_count"}}),
                        None,
                    )
                    if v_col:
                        st.subheader("Top Performing Videos")
                        n = st.slider("Number of videos", 5, 50, 15)
                        top = videos_df.nlargest(n, v_col)
                        title_col = next(
                            (c for c in top.columns if c.lower() in {{"title"}}),
                            None,
                        )
                        ch_col = next(
                            (c for c in top.columns if c.lower() in {{"channel_name", "channel_title", "channel"}}),
                            None,
                        )
                        display_cols = [c for c in [title_col, ch_col, v_col] if c]
                        st.dataframe(top[display_cols].reset_index(drop=True))

            elif section == "Audience Sentiment":
                st.title("Audience Sentiment")
                if comments_df.empty:
                    st.info("No comment data loaded. Run NLP pipeline first.")
                else:
                    sent_col = next(
                        (c for c in comments_df.columns if c.lower() == "sentiment_label"),
                        None,
                    )
                    if sent_col:
                        st.subheader("Sentiment Distribution")
                        counts = comments_df[sent_col].value_counts()
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.pie(
                            counts.values,
                            labels=counts.index,
                            autopct="%1.1f%%",
                            colors=["#2ca02c", "#d62728", "#7f7f7f"],
                        )
                        ax.set_title("Overall Sentiment")
                        st.pyplot(fig)

                    topic_col = next(
                        (c for c in comments_df.columns if c.lower() == "topic_label"),
                        None,
                    )
                    if topic_col:
                        st.subheader("Topic Distribution")
                        topic_counts = comments_df[topic_col].value_counts().head(15)
                        st.bar_chart(topic_counts)

            elif section == "Topic Trends":
                st.title("Topic Trends")
                if comments_df.empty:
                    st.info("No comment data loaded.")
                else:
                    st.info(
                        "Topic trends require temporal aggregation of comment topics. "
                        "Run the full NLP pipeline to generate topic_timeseries data."
                    )

            elif section == "Graph Influence Map":
                st.title("Graph Influence Map")
                st.info(
                    "Graph visualizations available after running the graph analysis module. "
                    "Load graph_stats.parquet and community data to populate this section."
                )

            elif section == "Strategic Recommendations":
                st.title("Strategic Recommendations")
                st.markdown(
                    """
                    ### Key Insights

                    Based on the data analysis across channels, the following strategic
                    recommendations emerge:

                    1. **Peak Posting Times**: Analyze the posting time heatmap to identify
                       optimal upload schedules for maximum engagement.

                    2. **Content Mix Optimization**: Leverage topic trends to adjust the
                       content mix — increase investment in emerging topics while
                       maintaining coverage of stable core topics.

                    3. **Sentiment Management**: Monitor negative sentiment trends and
                       proactively address audience concerns through transparent
                       communication.

                    4. **Cross-Channel Synergies**: Identify collaborative opportunities
                       between channels based on shared audience segments and topic overlaps.

                    5. **Engagement Quality**: Focus on building meaningful discussion
                       (comment depth) rather than vanity metrics alone.
                    """
                )

            # ------------------------------------------------------------------
            # Footer
            # ------------------------------------------------------------------
            st.markdown("---")
            st.caption(
                "Digital Media Analytics Dashboard — Auto-generated "
                f"by VisualizationEngine | Data dir: {{DATA_DIR}}"
            )
        ''')

        output_path = os.path.join(CONFIG["FIGURES_DIR"], "..", "streamlit_dashboard.py")
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            f.write(script)

        logger.info("Streamlit dashboard script saved → %s", output_path)
        return output_path

    # ==================================================================
    # 9. WORD CLOUD
    # ==================================================================

    def plot_wordcloud(
        self,
        texts: List[str],
        title: str = "Word Cloud",
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Generate and plot a word cloud from a list of texts."""
        if not texts:
            logger.warning("plot_wordcloud: empty text list.")
            return None

        cleaned_texts = [str(t) for t in texts if str(t).strip()]
        if not cleaned_texts:
            return None

        combined = " ".join(cleaned_texts)

        fig, ax = plt.subplots(figsize=(12, 8))

        try:
            from wordcloud import WordCloud

            wc = WordCloud(
                width=1200,
                height=800,
                background_color="white",
                max_words=200,
                colormap="viridis",
                collocations=False,
                random_state=CONFIG["RANDOM_STATE"],
            ).generate(combined)

            ax.imshow(wc, interpolation="bilinear")
            ax.set_axis_off()
            ax.set_title(title, fontsize=16, pad=20)
        except ImportError:
            logger.warning(
                "wordcloud library not available; falling back to frequency bar chart. "
                "Install with: pip install wordcloud"
            )
            # Frequency fallback
            from collections import Counter

            words = combined.lower().split()
            # Basic stopwords
            stop_words = {
                "the", "is", "in", "and", "to", "a", "of", "for", "it", "on",
                "that", "was", "with", "this", "are", "be", "have", "has", "as",
                "at", "by", "an", "or", "not", "from", "but", "we", "they",
            }
            words = [w for w in words if w not in stop_words and len(w) > 2]
            word_counts = Counter(words).most_common(40)
            if word_counts:
                w_words, w_counts = zip(*word_counts)
                ax.barh(
                    list(reversed(w_words)),
                    list(reversed(w_counts)),
                    color="#1f77b4",
                    edgecolor="black",
                    linewidth=0.3,
                )
                ax.set_title(title + " (Top Words)", fontsize=14)
                ax.set_xlabel("Frequency")

        if save:
            self.save_figure(fig, "wordcloud.png")
        return fig

    # ==================================================================
    # 10. ORCHESTRATION
    # ==================================================================

    def create_report_visualizations(self) -> Dict[str, str]:
        """Placeholder for comprehensive report generation.

        Returns a dict of section name → output path.
        """
        logger.info("create_report_visualizations: generating report placeholders.")
        results: Dict[str, str] = {}
        for section in [
            "overview",
            "temporal",
            "sentiment",
            "models",
            "recommendations",
        ]:
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.text(
                0.5,
                0.5,
                f"Report Section: {section.title()}\nGenerated automatically",
                ha="center",
                va="center",
                fontsize=14,
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            path = self.save_figure(fig, f"report_{section}.png")
            results[section] = path
        return results

    def generate_all_visualizations(
        self,
        videos_df: pd.DataFrame,
        comments_df: pd.DataFrame,
        modeling_results: Optional[Dict[str, Any]] = None,
        graph_results: Optional[Dict[str, Any]] = None,
        nlp_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Orchestrate creation of all visualizations.

        Parameters
        ----------
        videos_df : pd.DataFrame
            Processed videos data.
        comments_df : pd.DataFrame
            Processed comments data (ideally with NLP columns).
        modeling_results : dict or None
            Optional dict with keys: importance_df, cm, class_names, fpr_dict,
            tpr_dict, roc_auc_dict, y_true, y_pred, metrics_dict.
        graph_results : dict or None
            Optional dict with keys: graph_stats, central_nodes, communities,
            community_stats.
        nlp_results : dict or None
            Optional dict with keys: embeddings, labels, topic_timeseries,
            topic_data.

        Returns
        -------
        dict
            Mapping of visualization name → saved file path.
        """
        saved: Dict[str, str] = {}

        logger.info("=" * 60)
        logger.info("Generating all visualizations...")
        logger.info("=" * 60)

        # --- Statistical ---
        for method_name in [
            "plot_views_distribution",
            "plot_likes_distribution",
            "plot_comments_distribution",
            "plot_engagement_comparison",
            "plot_correlation_heatmap",
        ]:
            try:
                fig = getattr(self, method_name)(videos_df, save=True)
                if fig:
                    saved[method_name] = f"{method_name}.png"
            except Exception as exc:
                logger.warning("%s failed: %s", method_name, exc)

        for metric in ["views", "likes", "comments", "engagement_rate"]:
            try:
                fig = self.plot_top_performing_videos(videos_df, metric=metric, n=15, save=True)
                if fig:
                    saved[f"plot_top_performing_videos_{metric}"] = f"top_15_{metric}_videos.png"
            except Exception as exc:
                logger.warning("plot_top_performing_videos(%s) failed: %s", metric, exc)

        try:
            fig = self.plot_scatter_matrix(videos_df, save=True)
            if fig:
                saved["plot_scatter_matrix"] = "scatter_matrix.png"
        except Exception as exc:
            logger.warning("plot_scatter_matrix failed: %s", exc)

        try:
            fig = self.plot_anomaly_scatter(videos_df, save=True)
            if fig:
                saved["plot_anomaly_scatter"] = "anomaly_scatter.png"
        except Exception as exc:
            logger.warning("plot_anomaly_scatter failed: %s", exc)

        # --- Temporal ---
        temporal_methods = [
            "plot_engagement_timeline",
            "plot_posting_time_heatmap",
            "plot_day_of_week_effect",
            "plot_video_age_vs_views",
            "plot_comment_volume_over_time",
            "plot_engagement_decomposition",
        ]
        for method_name in temporal_methods:
            try:
                df_arg = comments_df if "comment_volume" in method_name else videos_df
                fig = getattr(self, method_name)(df_arg, save=True)
                if fig:
                    saved[method_name] = f"{method_name}.png"
            except Exception as exc:
                logger.warning("%s failed: %s", method_name, exc)

        # --- Sentiment & NLP ---
        try:
            fig = self.plot_sentiment_distribution(comments_df, save=True)
            if fig:
                saved["plot_sentiment_distribution"] = "sentiment_distribution.png"
        except Exception as exc:
            logger.warning("plot_sentiment_distribution failed: %s", exc)

        try:
            fig = self.plot_sentiment_by_channel(videos_df, comments_df, save=True)
            if fig:
                saved["plot_sentiment_by_channel"] = "sentiment_by_channel.png"
        except Exception as exc:
            logger.warning("plot_sentiment_by_channel failed: %s", exc)

        try:
            fig = self.plot_sentiment_timeline(comments_df, save=True)
            if fig:
                saved["plot_sentiment_timeline"] = "sentiment_timeline.png"
        except Exception as exc:
            logger.warning("plot_sentiment_timeline failed: %s", exc)

        try:
            fig = self.plot_topic_distribution(comments_df, save=True)
            if fig:
                saved["plot_topic_distribution"] = "topic_distribution.png"
        except Exception as exc:
            logger.warning("plot_topic_distribution failed: %s", exc)

        if nlp_results:
            topic_ts = nlp_results.get("topic_timeseries")
            if topic_ts is not None and not (
                isinstance(topic_ts, pd.DataFrame) and topic_ts.empty
            ):
                try:
                    fig = self.plot_topic_trends(topic_ts, save=True)
                    if fig:
                        saved["plot_topic_trends"] = "topic_trends.png"
                except Exception as exc:
                    logger.warning("plot_topic_trends failed: %s", exc)

            embeddings = nlp_results.get("embeddings")
            labels = nlp_results.get("labels")
            if embeddings is not None and labels is not None:
                try:
                    fig = self.plot_embedding_projection(
                        embeddings, labels, method="pca", save=True
                    )
                    if fig:
                        saved["plot_embedding_projection_pca"] = "embedding_projection_pca.png"
                except Exception as exc:
                    logger.warning("plot_embedding_projection(pca) failed: %s", exc)

                try:
                    fig = self.plot_embedding_projection(
                        embeddings, labels, method="umap", save=True
                    )
                    if fig:
                        saved["plot_embedding_projection_umap"] = "embedding_projection_umap.png"
                except Exception as exc:
                    logger.warning("plot_embedding_projection(umap) failed: %s", exc)

        # --- Wordcloud ---
        text_col = _resolve_col(
            comments_df, ["comment_text", "text_display", "text_cleaned", "text"]
        )
        if text_col:
            try:
                fig = self.plot_wordcloud(
                    comments_df[text_col].dropna().tolist(), save=True
                )
                if fig:
                    saved["plot_wordcloud"] = "wordcloud.png"
            except Exception as exc:
                logger.warning("plot_wordcloud failed: %s", exc)

        # --- Model ---
        if modeling_results:
            for key, method_name in [
                ("importance_df", "plot_feature_importance"),
                ("cm", "plot_confusion_matrix"),
                ("roc_data", "plot_roc_curves"),
                ("metrics_dict", "plot_model_comparison"),
            ]:
                try:
                    if key == "importance_df":
                        imp = modeling_results.get("importance_df")
                        if imp is not None:
                            fig = self.plot_feature_importance(imp, save=True)
                            if fig:
                                saved[method_name] = "feature_importance.png"
                    elif key == "cm":
                        cm_data = modeling_results.get("cm")
                        cnames = modeling_results.get("class_names", [])
                        if cm_data is not None:
                            fig = self.plot_confusion_matrix(cm_data, cnames, save=True)
                            if fig:
                                saved[method_name] = "confusion_matrix.png"
                    elif key == "roc_data":
                        fpr = modeling_results.get("fpr_dict")
                        tpr = modeling_results.get("tpr_dict")
                        auc = modeling_results.get("roc_auc_dict")
                        if fpr and tpr and auc:
                            fig = self.plot_roc_curves(fpr, tpr, auc, save=True)
                            if fig:
                                saved[method_name] = "roc_curves.png"
                    elif key == "metrics_dict":
                        mdict = modeling_results.get("metrics_dict")
                        if mdict:
                            fig = self.plot_model_comparison(mdict, save=True)
                            if fig:
                                saved[method_name] = "model_comparison.png"
                except Exception as exc:
                    logger.warning("%s failed: %s", method_name, exc)

            y_true = modeling_results.get("y_true")
            y_pred = modeling_results.get("y_pred")
            if y_true is not None and y_pred is not None:
                try:
                    fig = self.plot_residuals(y_true, y_pred, save=True)
                    if fig:
                        saved["plot_residuals"] = "residuals.png"
                except Exception as exc:
                    logger.warning("plot_residuals failed: %s", exc)
                try:
                    fig = self.plot_predictions_vs_actual(y_true, y_pred, save=True)
                    if fig:
                        saved["plot_predictions_vs_actual"] = "predictions_vs_actual.png"
                except Exception as exc:
                    logger.warning("plot_predictions_vs_actual failed: %s", exc)

        # --- Graph ---
        if graph_results:
            for key, method_name in [
                ("graph_stats", "plot_graph_summary"),
                ("central_nodes", "plot_centrality_distribution"),
                ("communities", "plot_community_sizes"),
                ("community_stats", "plot_community_composition"),
            ]:
                try:
                    val = graph_results.get(key)
                    if val is not None:
                        fig = getattr(self, method_name)(val, save=True)
                        if fig:
                            saved[method_name] = f"{method_name}.png"
                except Exception as exc:
                    logger.warning("%s failed: %s", method_name, exc)

        # --- Interactive Plotly ---
        try:
            self.interactive_engagement_scatter(videos_df, save=True)
            saved["interactive_engagement_scatter"] = "interactive_engagement_scatter.html"
        except Exception as exc:
            logger.warning("interactive_engagement_scatter failed: %s", exc)

        try:
            self.interactive_timeline(videos_df, save=True)
            saved["interactive_timeline"] = "interactive_timeline.html"
        except Exception as exc:
            logger.warning("interactive_timeline failed: %s", exc)

        if nlp_results:
            topic_data = nlp_results.get("topic_data") or nlp_results.get("topic_distribution")
            if topic_data is not None:
                try:
                    self.interactive_topic_explorer(topic_data, save=True)
                    saved["interactive_topic_explorer"] = "interactive_topic_explorer.html"
                except Exception as exc:
                    logger.warning("interactive_topic_explorer failed: %s", exc)

        try:
            self.interactive_channel_dashboard(videos_df, comments_df, save=True)
            saved["interactive_channel_dashboard"] = "channel_dashboard.html"
        except Exception as exc:
            logger.warning("interactive_channel_dashboard failed: %s", exc)

        logger.info("All visualizations complete. Saved %d files.", len(saved))
        logger.info("=" * 60)
        return saved


# ======================================================================
# __main__
# ======================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    data_dir = CONFIG["PROCESSED_DATA_DIR"]

    # Try to load processed data
    videos_path = os.path.join(data_dir, "videos_processed.parquet")
    comments_path = os.path.join(data_dir, "comments_processed.parquet")

    try:
        videos_df = pd.read_parquet(videos_path)
        logger.info("Loaded videos_processed: %s rows", len(videos_df))
    except FileNotFoundError:
        logger.warning("%s not found; trying videos.parquet", videos_path)
        try:
            videos_df = pd.read_parquet(os.path.join(data_dir, "videos.parquet"))
            logger.info("Loaded videos.parquet: %s rows", len(videos_df))
        except FileNotFoundError:
            logger.warning("No videos data found; generating demo data.")
            rng = np.random.default_rng(CONFIG["RANDOM_STATE"])
            n = 200
            channels = ["Aaj TV (Aaj News)", "Hum TV", "Raftar"]
            videos_df = pd.DataFrame(
                {
                    "video_id": [f"vid_{i:04d}" for i in range(n)],
                    "channel_name": rng.choice(channels, n),
                    "title": [f"Sample Video {i}" for i in range(n)],
                    "published_at": pd.to_datetime(
                        rng.integers(
                            pd.Timestamp("2024-05-01").value // 10**9,
                            pd.Timestamp("2026-05-01").value // 10**9,
                            size=n,
                        ),
                        unit="s",
                    ),
                    "views": rng.lognormal(mean=10, sigma=1.5, size=n).astype(int),
                    "likes": rng.lognormal(mean=7, sigma=1.8, size=n).astype(int),
                    "comments": rng.lognormal(mean=5, sigma=2.0, size=n).astype(int),
                    "video_age_days": rng.uniform(0, 365, size=n),
                }
            )
            videos_df["likes"] = np.minimum(videos_df["likes"], videos_df["views"])
            videos_df["comments"] = np.minimum(videos_df["comments"], videos_df["views"])
            videos_df["engagement_rate"] = np.where(
                videos_df["views"] > 0,
                (videos_df["likes"] + videos_df["comments"]) / videos_df["views"],
                0.0,
            )
            # Add temporal features
            videos_df["hour"] = videos_df["published_at"].dt.hour
            videos_df["day_of_week"] = videos_df["published_at"].dt.dayofweek
            videos_df["month"] = videos_df["published_at"].dt.month
            videos_df["year"] = videos_df["published_at"].dt.year
            videos_df["is_weekend"] = videos_df["day_of_week"].isin([5, 6]).astype(int)
            # Add outlier labels
            from sklearn.ensemble import IsolationForest

            iso = IsolationForest(contamination=0.05, random_state=CONFIG["RANDOM_STATE"])
            feat_cols = ["views", "likes", "comments"]
            features = videos_df[feat_cols].copy()
            features["views_log"] = np.log1p(features["views"])
            features["likes_log"] = np.log1p(features["likes"])
            features["comments_log"] = np.log1p(features["comments"])
            videos_df["outlier_label"] = (iso.fit_predict(features) == -1).astype(int)

    try:
        comments_df = pd.read_parquet(comments_path)
        logger.info("Loaded comments_processed: %s rows", len(comments_df))
    except FileNotFoundError:
        logger.warning("%s not found; trying comments.parquet", comments_path)
        try:
            comments_df = pd.read_parquet(os.path.join(data_dir, "comments.parquet"))
            logger.info("Loaded comments.parquet: %s rows", len(comments_df))
        except FileNotFoundError:
            logger.warning("No comments data found; generating demo data.")
            rng = np.random.default_rng(CONFIG["RANDOM_STATE"])
            n_comments = 500
            sent_labels = rng.choice(
                ["positive", "neutral", "negative"], n_comments, p=[0.45, 0.35, 0.2]
            )
            comments_df = pd.DataFrame(
                {
                    "comment_id": [f"cmt_{i:05d}" for i in range(n_comments)],
                    "video_id": rng.choice(
                        videos_df["video_id"].values, n_comments
                    ),
                    "comment_text": [
                        f"This is comment number {i}. "
                        + rng.choice(
                            [
                                "Great video!",
                                "Interesting perspective.",
                                "Not sure about this.",
                                "Loved the analysis.",
                                "Could be better.",
                                "Amazing content!",
                                "Share more like this.",
                            ]
                        )
                        for i in range(n_comments)
                    ],
                    "published_at": pd.to_datetime(
                        rng.integers(
                            pd.Timestamp("2024-05-01").value // 10**9,
                            pd.Timestamp("2026-05-01").value // 10**9,
                            size=n_comments,
                        ),
                        unit="s",
                    ),
                    "sentiment_label": sent_labels,
                    "sentiment_score": np.where(
                        sent_labels == "positive",
                        rng.uniform(0.1, 1.0, n_comments),
                        np.where(
                            sent_labels == "negative",
                            rng.uniform(-1.0, -0.1, n_comments),
                            rng.uniform(-0.1, 0.1, n_comments),
                        ),
                    ),
                    "topic_label": rng.choice(
                        [
                            "Politics",
                            "Entertainment",
                            "Music",
                            "Sports",
                            "Technology",
                            "Education",
                            "Health",
                            "Business",
                        ],
                        n_comments,
                    ),
                }
            )

    # Run visualization engine
    engine = VisualizationEngine(figsize=(12, 6), dpi=100)

    logger.info("Generating comprehensive visualizations...")
    saved = engine.generate_all_visualizations(
        videos_df=videos_df,
        comments_df=comments_df,
        modeling_results=None,
        graph_results=None,
        nlp_results=None,
    )

    logger.info("\nSaved visualizations:")
    for name, path in sorted(saved.items()):
        logger.info("  %-50s → %s", name, path)

    # Generate Streamlit dashboard
    dashboard_path = engine.create_streamlit_dashboard_code()
    logger.info("\nStreamlit dashboard: %s", dashboard_path)
    logger.info("Run with: streamlit run %s", dashboard_path)
