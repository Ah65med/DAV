"""
Graph Analysis module for Digital Media Analytics.

Builds a heterogeneous NetworkX graph from YouTube channel, video, comment,
user, and topic data. Provides centrality analysis, community detection,
link prediction, graph embeddings, an explainable analytics layer, and
graph export utilities.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import community as nx_community
from tqdm import tqdm

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG: Dict[str, Any] = {
    "RANDOM_STATE": 42,
    "GRAPHS_DIR": "./outputs/graphs",
    "PROCESSED_DATA_DIR": "./data/processed",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_norm(series: Union[pd.Series, np.ndarray, List[float]]) -> np.ndarray:
    """Min-max normalise a series, returning zeros if constant."""
    s = np.asarray(series, dtype=float)
    smin, smax = s.min(), s.max()
    if smax == smin:
        return np.zeros_like(s)
    return (s - smin) / (smax - smin)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# GraphAnalyzer
# ---------------------------------------------------------------------------


class GraphAnalyzer:
    """Builds and analyzes a heterogeneous graph over digital media entities.

    Parameters
    ----------
    random_state : int
        Seed for reproducible stochastic components.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.graph = nx.Graph()
        self._rng = np.random.default_rng(random_state)
        self._community_labels: Optional[Dict[Any, int]] = None

    # ==================================================================
    # Column resolution helper
    # ==================================================================

    @staticmethod
    def _col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # ==================================================================
    # 2. GRAPH CONSTRUCTION
    # ==================================================================

    def build_graph(
        self,
        channels_df: pd.DataFrame,
        videos_df: pd.DataFrame,
        comments_df: pd.DataFrame,
        users_df: Optional[pd.DataFrame] = None,
        topics_df: Optional[pd.DataFrame] = None,
    ) -> nx.Graph:
        """Build the heterogeneous graph from all entity DataFrames."""
        G = nx.Graph()

        # ---- column name resolution ----
        ch_id_col = self._col(channels_df, "channel_id")
        ch_name_col = self._col(channels_df, "channel_title", "channel_name")
        ch_video_count_col = self._col(channels_df, "video_count")
        ch_view_count_col = self._col(channels_df, "view_count")

        vid_id_col = self._col(videos_df, "video_id")
        vid_ch_col = self._col(videos_df, "channel_id")
        vid_title_col = self._col(videos_df, "title")
        vid_views_col = self._col(videos_df, "view_count", "views")
        vid_likes_col = self._col(videos_df, "like_count", "likes")
        vid_comments_col = self._col(videos_df, "comment_count", "comments")
        vid_eng_col = "engagement_rate"
        vid_pub_col = self._col(videos_df, "published_at")
        vid_success_col = self._col(videos_df, "success_label")

        cmt_id_col = self._col(comments_df, "comment_id")
        cmt_vid_col = self._col(comments_df, "video_id")
        cmt_text_col = self._col(comments_df, "comment_text", "text_display", "text")
        cmt_sent_col = self._col(comments_df, "sentiment_score")
        cmt_like_col = self._col(comments_df, "like_count")
        cmt_author_col = self._col(comments_df, "author_name", "author_channel_id")

        # ---- CHANNEL nodes ----
        print("Building channel nodes...")
        for _, row in channels_df.iterrows():
            cid = row[ch_id_col]
            attrs: Dict[str, Any] = {"type": "channel"}
            if ch_name_col:
                attrs["name"] = str(row[ch_name_col])
            if vid_ch_col and vid_ch_col in videos_df.columns and vid_eng_col in videos_df.columns:
                mask = videos_df[vid_ch_col] == cid
                attrs["avg_engagement"] = float(videos_df.loc[mask, vid_eng_col].mean()) if mask.any() else 0.0
            else:
                attrs["avg_engagement"] = 0.0
            if ch_video_count_col:
                attrs["total_videos"] = int(row[ch_video_count_col])
            if ch_view_count_col:
                attrs["total_views"] = int(row[ch_view_count_col])
            G.add_node(cid, **attrs)

        # ---- VIDEO nodes ----
        print("Building video nodes...")
        for _, row in tqdm(videos_df.iterrows(), total=len(videos_df), desc="  Videos", file=sys.stderr):
            vid = row[vid_id_col]
            attrs = {"type": "video"}
            if vid_title_col:
                attrs["title"] = str(row[vid_title_col])
            if vid_views_col:
                attrs["views"] = int(row[vid_views_col])
            if vid_likes_col:
                attrs["likes"] = int(row[vid_likes_col])
            if vid_comments_col:
                attrs["comments"] = int(row[vid_comments_col])
            if vid_eng_col in videos_df.columns:
                attrs["engagement_rate"] = float(row[vid_eng_col])
            if vid_pub_col:
                attrs["published_at"] = str(row[vid_pub_col])
            if vid_success_col:
                attrs["success_label"] = row[vid_success_col]
            G.add_node(vid, **attrs)

        # ---- USER nodes ----
        print("Building user nodes...")
        if users_df is not None and not users_df.empty:
            u_id_col = self._col(users_df, "user_id", "author_name", "author_channel_id")
            u_total_col = self._col(users_df, "total_comments")
            u_avg_sent_col = self._col(users_df, "avg_sentiment")
            u_inf_col = self._col(users_df, "influence_score")
            for _, row in users_df.iterrows():
                uid = row[u_id_col]
                attrs = {"type": "user"}
                if u_total_col:
                    attrs["total_comments"] = int(row[u_total_col])
                if u_avg_sent_col:
                    attrs["avg_sentiment"] = float(row[u_avg_sent_col])
                if u_inf_col:
                    attrs["influence_score"] = float(row[u_inf_col])
                G.add_node(uid, **attrs)
        else:
            print("  Building user nodes from comment authors...")
            author_counts = comments_df[cmt_author_col].value_counts()
            author_sentiment = comments_df.groupby(cmt_author_col)[cmt_sent_col].mean() if cmt_sent_col else None
            for author, count in author_counts.items():
                attrs = {
                    "type": "user",
                    "total_comments": int(count),
                }
                if author_sentiment is not None and author in author_sentiment.index:
                    attrs["avg_sentiment"] = float(author_sentiment[author])
                else:
                    attrs["avg_sentiment"] = 0.0
                attrs["influence_score"] = float(np.log1p(count))
                G.add_node(author, **attrs)

        # ---- COMMENT nodes ----
        print("Building comment nodes...")
        for _, row in tqdm(comments_df.iterrows(), total=len(comments_df), desc="  Comments", file=sys.stderr):
            cid = row[cmt_id_col]
            attrs = {"type": "comment"}
            if cmt_text_col:
                text = str(row[cmt_text_col])
                attrs["text"] = text[:200] if len(text) > 200 else text
            if cmt_sent_col:
                attrs["sentiment_score"] = float(row[cmt_sent_col])
            if cmt_like_col:
                attrs["like_count"] = int(row[cmt_like_col])
            G.add_node(cid, **attrs)

        # ---- TOPIC nodes ----
        print("Building topic nodes...")
        if topics_df is not None and not topics_df.empty:
            t_label_col = self._col(topics_df, "topic_label", "label")
            t_freq_col = self._col(topics_df, "frequency", "count", "proportion")
            t_avg_eng_col = self._col(topics_df, "avg_engagement")
            t_sent_col = self._col(topics_df, "sentiment_score", "mean_sentiment")
            for _, row in topics_df.iterrows():
                tlabel = str(row[t_label_col])
                attrs = {"type": "topic"}
                attrs["label"] = tlabel
                if t_freq_col:
                    attrs["frequency"] = float(row[t_freq_col])
                else:
                    attrs["frequency"] = 1.0
                if t_avg_eng_col:
                    attrs["avg_engagement"] = float(row[t_avg_eng_col])
                else:
                    attrs["avg_engagement"] = 0.0
                if t_sent_col:
                    attrs["sentiment_score"] = float(row[t_sent_col])
                G.add_node(tlabel, **attrs)

        # ---- EDGES ----
        # Channel -> Video (uploads)
        print("Building edges: Channel -> Video...")
        if vid_ch_col and vid_ch_col in videos_df.columns:
            for _, row in tqdm(videos_df.iterrows(), total=len(videos_df), desc="  Ch->Vid", file=sys.stderr):
                weight = float(row.get(vid_eng_col, 0.01)) if vid_eng_col in videos_df.columns else 0.01
                weight = max(1e-6, weight)
                G.add_edge(row[vid_ch_col], row[vid_id_col], relation="uploads", weight=weight)

        # Video -> Comment (receives)
        print("Building edges: Video -> Comment...")
        if cmt_vid_col and cmt_vid_col in comments_df.columns:
            for _, row in tqdm(comments_df.iterrows(), total=len(comments_df), desc="  Vid->Cmt", file=sys.stderr):
                like_w = float(row.get(cmt_like_col, 0)) if cmt_like_col else 0
                weight = max(1e-6, np.log1p(like_w))
                G.add_edge(row[cmt_vid_col], row[cmt_id_col], relation="receives", weight=weight)

        # User -> Comment (writes)
        print("Building edges: User -> Comment...")
        if cmt_author_col and cmt_author_col in comments_df.columns:
            for _, row in tqdm(comments_df.iterrows(), total=len(comments_df), desc="  Usr->Cmt", file=sys.stderr):
                user_id = row[cmt_author_col]
                if G.has_node(user_id):
                    G.add_edge(user_id, row[cmt_id_col], relation="writes", weight=1.0)

        # Comment -> Topic (belongs_to)
        print("Building edges: Comment -> Topic...")
        topic_label_col = self._col(comments_df, "topic_label")
        if topic_label_col and topic_label_col in comments_df.columns:
            for _, row in tqdm(comments_df.iterrows(), total=len(comments_df), desc="  Cmt->Topic", file=sys.stderr):
                tlabel = str(row[topic_label_col])
                if tlabel != "noise" and G.has_node(tlabel):
                    G.add_edge(row[cmt_id_col], tlabel, relation="belongs_to", weight=1.0)

        # Video -> Topic (discusses)
        if topic_label_col and topic_label_col in comments_df.columns and vid_id_col:
            print("Building edges: Video -> Topic...")
            vid_topic_counts: Dict[Tuple[Any, str], int] = defaultdict(int)
            for _, row in comments_df.iterrows():
                tlabel = str(row[topic_label_col])
                if tlabel == "noise" or not G.has_node(tlabel):
                    continue
                vid_topic_counts[(row[cmt_vid_col], tlabel)] += 1
            for (vid, tlabel), cnt in vid_topic_counts.items():
                G.add_edge(vid, tlabel, relation="discusses", weight=float(cnt))

        # User -> Video (engages_with)
        print("Building edges: User -> Video...")
        if cmt_author_col and cmt_vid_col:
            uv_counts: Dict[Tuple[Any, Any], int] = defaultdict(int)
            for _, row in comments_df.iterrows():
                user_id = row[cmt_author_col]
                vid = row[cmt_vid_col]
                if G.has_node(user_id) and G.has_node(vid):
                    uv_counts[(user_id, vid)] += 1
            for (user_id, vid), cnt in uv_counts.items():
                G.add_edge(user_id, vid, relation="engages_with", weight=float(np.log1p(cnt)))

        # Topic <-> Topic (semantic_similarity)
        if topic_label_col and topic_label_col in comments_df.columns:
            print("Building edges: Topic <-> Topic...")
            topic_comment_sets: Dict[str, set] = defaultdict(set)
            for _, row in comments_df.iterrows():
                tlabel = str(row[topic_label_col])
                if tlabel != "noise" and G.has_node(tlabel):
                    topic_comment_sets[tlabel].add(row[cmt_vid_col])
            topics_list = list(topic_comment_sets.keys())
            for i in range(len(topics_list)):
                for j in range(i + 1, len(topics_list)):
                    ta, tb = topics_list[i], topics_list[j]
                    set_a = topic_comment_sets[ta]
                    set_b = topic_comment_sets[tb]
                    if set_a and set_b:
                        intersection = len(set_a & set_b)
                        union = len(set_a | set_b)
                        overlap = intersection / union if union > 0 else 0.0
                        if overlap > 0:
                            G.add_edge(ta, tb, relation="semantic_similarity", weight=overlap)

        self.graph = G
        print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def add_video_similarity_edges(
        self, video_embeddings: Optional[np.ndarray] = None, threshold: float = 0.7
    ) -> int:
        """Add semantic-similarity edges between videos.

        Uses cosine similarity of embeddings when available; falls back to
        numerical-attribute similarity.
        """
        G = self.graph
        video_nodes = [n for n, a in G.nodes(data=True) if a.get("type") == "video"]
        n_videos = len(video_nodes)
        if n_videos < 2:
            return 0

        added = 0

        if video_embeddings is not None and video_embeddings.shape[0] >= n_videos:
            print(f"Computing video semantic similarity (threshold={threshold})...")
            for i in tqdm(range(n_videos), desc="  Video<->Video", file=sys.stderr):
                for j in range(i + 1, n_videos):
                    sim = _cosine_similarity(video_embeddings[i], video_embeddings[j])
                    if sim >= threshold:
                        G.add_edge(video_nodes[i], video_nodes[j],
                                   relation="semantic_similarity", weight=float(sim))
                        added += 1
        else:
            print("No embeddings provided; using numerical-attribute similarity fallback...")
            features: Dict[Any, np.ndarray] = {}
            for v in video_nodes:
                attrs = G.nodes[v]
                fvec = np.array([
                    float(attrs.get("engagement_rate", 0)),
                    float(attrs.get("views", 0)),
                    float(attrs.get("likes", 0)),
                    float(attrs.get("comments", 0)),
                ], dtype=float)
                features[v] = fvec
            farr = np.array(list(features.values()))
            farr_norm = _safe_norm(farr.T).T
            for i in tqdm(range(n_videos), desc="  Video<->Video (fallback)", file=sys.stderr):
                for j in range(i + 1, n_videos):
                    sim = _cosine_similarity(farr_norm[i], farr_norm[j])
                    if sim >= threshold:
                        G.add_edge(video_nodes[i], video_nodes[j],
                                   relation="semantic_similarity", weight=float(sim))
                        added += 1

        print(f"  Added {added} video similarity edges")
        return added

    def build_bipartite_graph(
        self,
        type_a_nodes: List[Any],
        type_b_nodes: List[Any],
        edges_df: pd.DataFrame,
    ) -> nx.Graph:
        """Build a bipartite sub-graph between two sets of nodes."""
        B = nx.Graph()
        B.add_nodes_from(type_a_nodes, bipartite=0)
        B.add_nodes_from(type_b_nodes, bipartite=1)
        weight_col = "weight" if "weight" in edges_df.columns else None
        for _, row in edges_df.iterrows():
            src, tgt = row["source"], row["target"]
            w = float(row[weight_col]) if weight_col else 1.0
            if B.has_node(src) and B.has_node(tgt):
                B.add_edge(src, tgt, weight=w)
        return B

    def get_graph_stats(self) -> Dict[str, Any]:
        """Return key graph statistics: nodes, edges, density, components, avg degree."""
        G = self.graph
        n = G.number_of_nodes()
        m = G.number_of_edges()
        density = (2 * m) / (n * (n - 1)) if n > 1 else 0.0
        try:
            n_components = nx.number_connected_components(G)
        except Exception:
            n_components = 1
        degrees = [d for _, d in G.degree()]
        avg_degree = float(np.mean(degrees)) if degrees else 0.0
        return {
            "num_nodes": n,
            "num_edges": m,
            "density": density,
            "num_components": n_components,
            "avg_degree": avg_degree,
        }

    # ==================================================================
    # 3. CENTRALITY ANALYSIS
    # ==================================================================

    def compute_centrality_metrics(self) -> Dict[str, Dict[Any, float]]:
        """Compute and store centrality metrics as node attributes.

        Computes degree, betweenness, pagerank (weighted),
        eigenvector (with fallback), and closeness centrality.
        """
        G = self.graph
        print("Computing centrality metrics...")

        deg = nx.degree_centrality(G)
        nx.set_node_attributes(G, deg, "degree_centrality")

        print("  betweenness_centrality...")
        if G.number_of_nodes() > 5000:
            k = min(500, G.number_of_nodes())
            bet = nx.betweenness_centrality(G, k=k, weight="weight", seed=self.random_state)
        else:
            bet = nx.betweenness_centrality(G, weight="weight")
        nx.set_node_attributes(G, bet, "betweenness_centrality")

        print("  pagerank...")
        pr = nx.pagerank(G, weight="weight")
        nx.set_node_attributes(G, pr, "pagerank")

        print("  eigenvector_centrality...")
        try:
            eig = nx.eigenvector_centrality_numpy(G, weight="weight")
        except Exception:
            try:
                eig = nx.eigenvector_centrality(G, weight="weight", max_iter=500, tol=1e-4)
            except Exception:
                logger.warning("Eigenvector centrality failed; falling back to pagerank values.")
                eig = pr.copy()
        nx.set_node_attributes(G, eig, "eigenvector_centrality")

        print("  closeness_centrality...")
        if G.number_of_nodes() > 3000:
            try:
                clo = nx.closeness_centrality(G)
            except Exception:
                clo = {n: 0.0 for n in G.nodes()}
        else:
            clo = nx.closeness_centrality(G)
        nx.set_node_attributes(G, clo, "closeness_centrality")

        print("Centrality metrics computed.")
        return {"degree": deg, "betweenness": bet, "pagerank": pr, "eigenvector": eig, "closeness": clo}

    def get_top_central_nodes(
        self,
        centrality_metric: str = "pagerank",
        n: int = 20,
        node_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return top *n* nodes by a centrality metric, optionally filtered by type."""
        G = self.graph
        test_node = next(iter(G.nodes()), None)
        if test_node is not None and centrality_metric not in G.nodes[test_node]:
            self.compute_centrality_metrics()

        scored = []
        for node, attrs in G.nodes(data=True):
            if node_type and attrs.get("type") != node_type:
                continue
            score = attrs.get(centrality_metric, 0.0)
            scored.append((node, score, attrs.get("type", "unknown")))

        scored.sort(key=lambda x: x[1], reverse=True)
        return pd.DataFrame(scored[:n], columns=["node", centrality_metric, "type"])

    def get_influential_videos(self, n: int = 20) -> pd.DataFrame:
        """Top videos by pagerank."""
        return self.get_top_central_nodes("pagerank", n=n, node_type="video")

    def get_influential_users(self, n: int = 20) -> pd.DataFrame:
        """Top users by pagerank."""
        return self.get_top_central_nodes("pagerank", n=n, node_type="user")

    def get_influential_topics(self, n: int = 10) -> pd.DataFrame:
        """Top topics by betweenness centrality."""
        return self.get_top_central_nodes("betweenness_centrality", n=n, node_type="topic")

    # ==================================================================
    # 4. COMMUNITY DETECTION
    # ==================================================================

    def detect_communities_louvain(self) -> Dict[Any, int]:
        """Detect communities using Louvain (python-louvain).

        Falls back to greedy_modularity_communities if python-louvain is unavailable.
        Assigns ``community_id`` node attribute.
        """
        G = self.graph
        print("Detecting communities (Louvain)...")
        community_map: Dict[Any, int] = {}

        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G, weight="weight",
                                                          random_state=self.random_state)
            community_map = {n: int(c) for n, c in partition.items()}
            nx.set_node_attributes(G, community_map, "community_id")
            print(f"  Louvain found {len(set(community_map.values()))} communities")
        except ImportError:
            logger.warning("python-louvain not installed. Using greedy_modularity_communities.")
            comms = list(nx_community.greedy_modularity_communities(G, weight="weight"))
            for cid, node_set in enumerate(comms):
                for node in node_set:
                    community_map[node] = cid
            nx.set_node_attributes(G, community_map, "community_id")
            print(f"  Greedy found {len(comms)} communities")

        self._community_labels = community_map
        return community_map

    def get_community_stats(self) -> pd.DataFrame:
        """Return per-community statistics DataFrame."""
        G = self.graph
        if self._community_labels is None:
            self.detect_communities_louvain()

        comm_nodes: Dict[int, List[Any]] = defaultdict(list)
        for node, attrs in G.nodes(data=True):
            cid = attrs.get("community_id", -1)
            comm_nodes[cid].append(node)

        rows = []
        for cid, nodes in comm_nodes.items():
            n_size = len(nodes)
            types = [G.nodes[n].get("type", "unknown") for n in nodes]
            type_counts = Counter(types)
            dominant = type_counts.most_common(1)[0][0] if type_counts else "unknown"

            eng_vals = []
            for n in nodes:
                a = G.nodes[n]
                er = a.get("engagement_rate") or a.get("avg_engagement") or 0.0
                eng_vals.append(float(er))
            avg_eng = float(np.mean(eng_vals)) if eng_vals else 0.0

            pr_vals = [G.nodes[n].get("pagerank", 0.0) for n in nodes]
            avg_pr = float(np.mean(pr_vals)) if pr_vals else 0.0

            rows.append({
                "community_id": cid,
                "size": n_size,
                "dominant_node_type": dominant,
                "avg_engagement": avg_eng,
                "avg_pagerank": avg_pr,
                "node_type_distribution": dict(type_counts),
            })

        df = pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)
        return df

    def identify_community_themes(
        self, topic_assignments: Optional[pd.DataFrame] = None
    ) -> Dict[int, List[str]]:
        """Describe each community by its dominant topics."""
        G = self.graph
        if self._community_labels is None:
            self.detect_communities_louvain()

        comm_topics: Dict[int, List[str]] = defaultdict(list)

        for node, attrs in G.nodes(data=True):
            cid = attrs.get("community_id", -1)
            if attrs.get("type") == "topic":
                tlabel = attrs.get("label", str(node))
                comm_topics[cid].append(tlabel)
            elif attrs.get("type") == "comment":
                tlabel = attrs.get("topic_label")
                if tlabel:
                    comm_topics[cid].append(tlabel)

        result: Dict[int, List[str]] = {}
        for cid, topics in comm_topics.items():
            cnt = Counter(topics)
            result[cid] = [t for t, _ in cnt.most_common(10)]
        return result

    def detect_communities_girvan_newman(self, n_communities: int = 5) -> Dict[Any, int]:
        """Alternative community detection using Girvan-Newman.

        Simplified: iterates until the target number of communities is reached.
        Warns and subsamples for graphs with >2000 nodes.
        """
        G = self.graph
        print(f"Detecting communities (Girvan-Newman, target={n_communities})...")

        if G.number_of_nodes() > 2000:
            logger.warning("Graph too large for Girvan-Newman (>2000 nodes). Subsampling largest component.")
            components = sorted(nx.connected_components(G), key=len, reverse=True)
            sub = G.subgraph(components[0]).copy()
            sub = sub.subgraph(list(sub.nodes())[:2000])
        else:
            sub = G.copy()

        comp = nx_community.girvan_newman(sub)
        comm_set = None
        for communities in comp:
            if len(communities) >= n_communities:
                comm_set = communities
                break

        if comm_set is None:
            comm_set = list(nx.connected_components(sub))

        mapping: Dict[Any, int] = {}
        for cid, node_set in enumerate(comm_set):
            for node in node_set:
                mapping[node] = cid

        for node in G.nodes():
            if node not in mapping:
                mapping[node] = -1

        nx.set_node_attributes(G, mapping, "community_id")
        self._community_labels = mapping
        print(f"  Girvan-Newman found {len(set(mapping.values()))} communities")
        return mapping

    # ==================================================================
    # 5. LINK PREDICTION
    # ==================================================================

    def predict_links_common_neighbors(
        self, node_pairs: List[Tuple[Any, Any]]
    ) -> List[float]:
        """Common Neighbors score for each pair."""
        G = self.graph
        scores = []
        for u, v in node_pairs:
            preds = list(nx.common_neighbors(G, u, v))
            scores.append(float(len(preds)))
        return scores

    def predict_links_jaccard(
        self, node_pairs: List[Tuple[Any, Any]]
    ) -> List[float]:
        """Jaccard coefficient for each pair."""
        G = self.graph
        scores = []
        for u, v in node_pairs:
            if G.has_node(u) and G.has_node(v):
                preds = list(nx.jaccard_coefficient(G, [(u, v)]))
                scores.append(float(preds[0][2]) if preds else 0.0)
            else:
                scores.append(0.0)
        return scores

    def predict_links_preferential_attachment(
        self, node_pairs: List[Tuple[Any, Any]]
    ) -> List[float]:
        """Preferential attachment score for each pair."""
        G = self.graph
        scores = []
        for u, v in node_pairs:
            if G.has_node(u) and G.has_node(v):
                preds = list(nx.preferential_attachment(G, [(u, v)]))
                scores.append(float(preds[0][2]) if preds else 0.0)
            else:
                scores.append(0.0)
        return scores

    def predict_links_adamic_adar(
        self, node_pairs: List[Tuple[Any, Any]]
    ) -> List[float]:
        """Adamic-Adar index for each pair."""
        G = self.graph
        scores = []
        for u, v in node_pairs:
            if G.has_node(u) and G.has_node(v):
                preds = list(nx.adamic_adar_index(G, [(u, v)]))
                scores.append(float(preds[0][2]) if preds else 0.0)
            else:
                scores.append(0.0)
        return scores

    def rank_link_predictions(
        self,
        node_type_a: str = "user",
        node_type_b: str = "video",
        top_n: int = 50,
    ) -> pd.DataFrame:
        """Rank all non-edge pairs between two node types by composite link-prediction score.

        Blends common neighbors, Jaccard, preferential attachment, and Adamic-Adar.
        """
        G = self.graph
        nodes_a = [n for n, a in G.nodes(data=True) if a.get("type") == node_type_a]
        nodes_b = [n for n, a in G.nodes(data=True) if a.get("type") == node_type_b]

        if len(nodes_a) * len(nodes_b) > 50_000:
            logger.warning("Candidate pool too large (%d x %d). Sampling.", len(nodes_a), len(nodes_b))
            nodes_a = list(self._rng.choice(nodes_a, size=min(500, len(nodes_a)), replace=False))
            nodes_b = list(self._rng.choice(nodes_b, size=min(500, len(nodes_b)), replace=False))

        print(f"Ranking link predictions ({len(nodes_a)} x {len(nodes_b)} candidates)...")
        pairs: List[Tuple[Any, Any]] = []
        for a in tqdm(nodes_a, desc="  Pairs", file=sys.stderr):
            a_neighbors = set(G.neighbors(a))
            for b in nodes_b:
                if b not in a_neighbors and a != b:
                    pairs.append((a, b))

        empty_df = pd.DataFrame(columns=[
            "source", "target", "common_neighbors", "jaccard",
            "preferential_attachment", "adamic_adar", "composite_score",
        ])
        if not pairs:
            return empty_df

        cn = self.predict_links_common_neighbors(pairs)
        jc = self.predict_links_jaccard(pairs)
        pa = self.predict_links_preferential_attachment(pairs)
        aa = self.predict_links_adamic_adar(pairs)

        cn_n = _safe_norm(cn)
        jc_n = _safe_norm(jc)
        pa_n = _safe_norm(pa)
        aa_n = _safe_norm(aa)

        rows = []
        for i, (a, b) in enumerate(pairs):
            composite = 0.25 * cn_n[i] + 0.25 * jc_n[i] + 0.25 * pa_n[i] + 0.25 * aa_n[i]
            rows.append({
                "source": a,
                "target": b,
                "common_neighbors": cn[i],
                "jaccard": jc[i],
                "preferential_attachment": pa[i],
                "adamic_adar": aa[i],
                "composite_score": composite,
            })

        df = pd.DataFrame(rows).sort_values("composite_score", ascending=False).head(top_n)
        return df.reset_index(drop=True)

    def evaluate_link_prediction(
        self,
        test_edges: List[Tuple[Any, Any]],
        pred_scores: List[float],
        k: int = 20,
    ) -> Dict[str, float]:
        """Compute precision@k for link prediction."""
        test_set = set(test_edges)
        if not test_set:
            return {"precision@k": 0.0, "avg_rank": 0.0, "n_test_edges": 0}

        ranked = sorted(zip(test_edges, pred_scores), key=lambda x: x[1], reverse=True)
        top_k_edges = set(e for e, _ in ranked[:k])
        hits = len(top_k_edges & test_set)
        precision_k = hits / min(k, len(test_set))

        ranks = []
        for i, (e, _) in enumerate(ranked):
            if e in test_set:
                ranks.append(i + 1)
        avg_rank = float(np.mean(ranks)) if ranks else float("inf")

        return {
            "precision@k": precision_k,
            "avg_rank": avg_rank,
            "n_test_edges": len(test_set),
        }

    # ==================================================================
    # 6. GRAPH EMBEDDINGS (Simplified)
    # ==================================================================

    def compute_node2vec_embeddings(
        self,
        dimensions: int = 64,
        walk_length: int = 30,
        num_walks: int = 200,
    ) -> Dict[Any, np.ndarray]:
        """Simplified Node2Vec using random walks + gensim Word2Vec.

        Falls back to feature aggregation if gensim is unavailable.
        """
        G = self.graph
        nodes = list(G.nodes())
        if len(nodes) < 3:
            logger.warning("Too few nodes for Node2Vec. Returning zero embeddings.")
            return {n: np.zeros(dimensions) for n in nodes}

        print("Computing simplified Node2Vec embeddings...")

        walks: List[List[Any]] = []
        for _ in tqdm(range(num_walks), desc="  Random walks", file=sys.stderr):
            for start_node in self._rng.choice(nodes, size=len(nodes), replace=False):
                if not list(G.neighbors(start_node)):
                    walks.append([start_node])
                    continue
                walk = [start_node]
                current = start_node
                for __ in range(walk_length - 1):
                    neighbors = list(G.neighbors(current))
                    if not neighbors:
                        break
                    current = neighbors[self._rng.integers(len(neighbors))]
                    walk.append(current)
                walks.append(walk)

        try:
            from gensim.models import Word2Vec
            str_walks = [[str(n) for n in w] for w in walks]
            model = Word2Vec(
                str_walks,
                vector_size=dimensions,
                window=5,
                min_count=1,
                workers=os.cpu_count() or 2,
                seed=self.random_state,
            )
            embeddings = {n: model.wv[str(n)] for n in nodes}
            print(f"  Node2Vec complete ({dimensions} dims).")
        except ImportError:
            logger.warning("gensim not installed. Falling back to aggregated features.")
            embeddings = self.aggregate_node_features(dimensions=dimensions)

        return embeddings

    def aggregate_node_features(
        self, node_type: Optional[str] = None, dimensions: int = 64
    ) -> Dict[Any, np.ndarray]:
        """Aggregate centrality + neighbor features into fixed-size embedding vectors.

        For each node constructs: [degree_cent, betweenness_cent, pagerank,
        eigenvector_cent, closeness_cent, degree, clustering_coeff,
        num_neighbors, avg_neighbor_degree, avg_neighbor_pagerank].
        """
        G = self.graph
        test_node = next(iter(G.nodes()), None)
        if test_node is not None and "pagerank" not in G.nodes[test_node]:
            self.compute_centrality_metrics()

        clustering = nx.clustering(G)
        avg_neighbor_degree = nx.average_neighbor_degree(G, weight="weight")

        embeddings: Dict[Any, np.ndarray] = {}
        nodes = [n for n, a in G.nodes(data=True) if node_type is None or a.get("type") == node_type]

        for node in nodes:
            attrs = G.nodes[node]
            neighbors = list(G.neighbors(node))
            neighbor_pr = [G.nodes[nb].get("pagerank", 0.0) for nb in neighbors]
            vec = np.array([
                float(attrs.get("degree_centrality", 0)),
                float(attrs.get("betweenness_centrality", 0)),
                float(attrs.get("pagerank", 0)),
                float(attrs.get("eigenvector_centrality", 0)),
                float(attrs.get("closeness_centrality", 0)),
                float(G.degree(node)),
                float(clustering.get(node, 0)),
                float(len(neighbors)),
                float(avg_neighbor_degree.get(node, 0)),
                float(np.mean(neighbor_pr)) if neighbor_pr else 0.0,
            ], dtype=float)
            if len(vec) < dimensions:
                vec = np.pad(vec, (0, dimensions - len(vec)))
            elif len(vec) > dimensions:
                vec = vec[:dimensions]
            embeddings[node] = vec

        return embeddings

    # ==================================================================
    # 7. GRAPH NEURAL NETWORK (Placeholder)
    # ==================================================================

    def build_gnn_node_classifier(self) -> None:
        """Placeholder for a full GNN-based node classifier.

        A production implementation would use PyTorch Geometric to:
        - Convert the NetworkX graph to a torch_geometric.data.Data object.
        - Build a GCN / GAT / GraphSAGE model.
        - Train on node labels (e.g. success_label for videos).
        - Evaluate with standard classification metrics.

        Use ``simplified_graph_classifier`` for a lightweight alternative.
        """
        logger.info(
            "build_gnn_node_classifier is a placeholder. "
            "Use simplified_graph_classifier for a lightweight alternative, "
            "or integrate PyTorch Geometric for a full GCN/GAT model."
        )

    def simplified_graph_classifier(
        self,
        target_node_type: str = "video",
        target_attr: str = "success_label",
        n_estimators: int = 100,
    ) -> Dict[str, Any]:
        """Lightweight RandomForest classifier using centrality + neighbor-aggregation features.

        Parameters
        ----------
        target_node_type : str
            Node type to classify (e.g. 'video').
        target_attr : str
            Node attribute to predict (must exist on nodes of target_node_type).
        n_estimators : int
            Number of trees in the RandomForest.

        Returns
        -------
        dict
            Keys: model, accuracy, classification_report, feature_importance, X, y, predictions.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import classification_report

        G = self.graph
        node_feats = self.aggregate_node_features(node_type=target_node_type)

        X_list = []
        y_list = []
        node_ids = []
        for node, feats in node_feats.items():
            val = G.nodes[node].get(target_attr)
            if val is None:
                continue
            X_list.append(feats)
            y_list.append(val)
            node_ids.append(node)

        if not X_list:
            logger.warning("No labelled nodes found for target_node_type='%s', target_attr='%s'.",
                           target_node_type, target_attr)
            return {}

        X = np.array(X_list)
        y = np.array(y_list)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(X, y)
        predictions = model.predict(X)
        accuracy = float(np.mean(predictions == y))
        report = classification_report(y, predictions)

        importances = model.feature_importances_
        feature_names = [
            "degree_centrality", "betweenness_centrality", "pagerank",
            "eigenvector_centrality", "closeness_centrality", "degree",
            "clustering_coeff", "num_neighbors", "avg_neighbor_degree",
            "avg_neighbor_pagerank",
        ][:X.shape[1]]
        fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)

        try:
            cv_scores = cross_val_score(model, X, y, cv=5)
            cv_mean = float(cv_scores.mean())
            cv_std = float(cv_scores.std())
        except Exception:
            cv_mean, cv_std = 0.0, 0.0

        return {
            "model": model,
            "accuracy": accuracy,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "classification_report": report,
            "feature_importance": fi_df,
            "X": X,
            "y": y,
            "predictions": predictions,
            "node_ids": node_ids,
        }

    # ==================================================================
    # 8. EXPLAINABLE GRAPH ANALYTICS
    # ==================================================================

    def explain_central_node(self, node_id: Any) -> Dict[str, Any]:
        """Return a detailed explanation of why a node is central.

        Includes: node attributes, centrality metrics, neighbour count,
        community assignment, top-connected node types.
        """
        G = self.graph
        if node_id not in G:
            return {"error": f"Node '{node_id}' not found in graph."}

        attrs = dict(G.nodes[node_id])
        neighbors = list(G.neighbors(node_id))
        neighbor_types = Counter(G.nodes[n].get("type", "unknown") for n in neighbors)

        return {
            "node_id": node_id,
            "node_type": attrs.get("type", "unknown"),
            "attributes": attrs,
            "degree": G.degree(node_id),
            "num_neighbors": len(neighbors),
            "neighbor_types": dict(neighbor_types),
            "centrality": {
                "degree_centrality": attrs.get("degree_centrality", 0),
                "betweenness_centrality": attrs.get("betweenness_centrality", 0),
                "pagerank": attrs.get("pagerank", 0),
                "eigenvector_centrality": attrs.get("eigenvector_centrality", 0),
                "closeness_centrality": attrs.get("closeness_centrality", 0),
            },
            "community_id": attrs.get("community_id", -1),
            "edge_relations": Counter(
                G.edges[node_id, n].get("relation", "unknown") for n in neighbors
            ),
        }

    def explain_community(self, community_id: int) -> Dict[str, Any]:
        """Explain the characteristics of a community."""
        G = self.graph
        nodes_in_comm = [n for n, a in G.nodes(data=True) if a.get("community_id") == community_id]
        if not nodes_in_comm:
            return {"error": f"No nodes found in community {community_id}."}

        types = Counter(G.nodes[n].get("type", "unknown") for n in nodes_in_comm)
        subgraph = G.subgraph(nodes_in_comm)
        internal_edges = subgraph.number_of_edges()
        external_edges = 0
        for n in nodes_in_comm:
            for nb in G.neighbors(n):
                if nb not in subgraph:
                    external_edges += 1
        external_edges //= 2

        eng_vals = [float(G.nodes[n].get("engagement_rate", G.nodes[n].get("avg_engagement", 0))) for n in nodes_in_comm]
        avg_eng = float(np.mean(eng_vals)) if eng_vals else 0.0

        themes = self.identify_community_themes()
        community_themes = themes.get(community_id, [])

        return {
            "community_id": community_id,
            "size": len(nodes_in_comm),
            "node_types": dict(types),
            "dominant_type": types.most_common(1)[0][0] if types else "unknown",
            "internal_edges": internal_edges,
            "external_edges": external_edges,
            "avg_engagement": avg_eng,
            "conductance": external_edges / (2 * internal_edges + external_edges) if (internal_edges + external_edges) > 0 else 1.0,
            "themes": community_themes,
        }

    def explain_link_prediction(
        self, node_a: Any, node_b: Any, score: Optional[float] = None
    ) -> Dict[str, Any]:
        """Explain why a link between node_a and node_b is predicted."""
        G = self.graph
        if not G.has_node(node_a):
            return {"error": f"Node '{node_a}' not in graph."}
        if not G.has_node(node_b):
            return {"error": f"Node '{node_b}' not in graph."}

        already_connected = G.has_edge(node_a, node_b)
        common_nb = list(nx.common_neighbors(G, node_a, node_b))

        explanation: Dict[str, Any] = {
            "node_a": {"id": node_a, "type": G.nodes[node_a].get("type"), "degree": G.degree(node_a)},
            "node_b": {"id": node_b, "type": G.nodes[node_b].get("type"), "degree": G.degree(node_b)},
            "already_connected": already_connected,
            "common_neighbors": len(common_nb),
            "common_neighbor_list": common_nb[:20],
            "shared_community": G.nodes[node_a].get("community_id") == G.nodes[node_b].get("community_id"),
        }

        preds = list(nx.jaccard_coefficient(G, [(node_a, node_b)]))
        if preds:
            explanation["jaccard"] = preds[0][2]
        preds = list(nx.preferential_attachment(G, [(node_a, node_b)]))
        if preds:
            explanation["preferential_attachment"] = preds[0][2]
        preds = list(nx.adamic_adar_index(G, [(node_a, node_b)]))
        if preds:
            explanation["adamic_adar"] = preds[0][2]
        if score is not None:
            explanation["prediction_score"] = score

        return explanation

    # ==================================================================
    # 9. GRAPH EXPORT
    # ==================================================================

    def export_to_pyvis(
        self,
        output_file: str = "graph.html",
        node_size_attr: str = "pagerank",
        color_by: str = "community",
    ) -> str:
        """Export interactive graph using PyVis.

        Parameters
        ----------
        output_file : str
            Path to output HTML file.
        node_size_attr : str
            Node attribute to scale node sizes (e.g. 'pagerank').
        color_by : str
            'community' to color by community_id, or a node attribute name.

        Returns
        -------
        str
            Path to the exported file.
        """
        G = self.graph

        try:
            from pyvis.network import Network
        except ImportError:
            logger.error("pyvis not installed. Install with: pip install pyvis")
            raise ImportError("pyvis is required for interactive graph export.")

        # Ensure centrality & communities exist
        test_node = next(iter(G.nodes()), None)
        if test_node is not None:
            if node_size_attr not in G.nodes[test_node]:
                self.compute_centrality_metrics()
            if color_by == "community" and "community_id" not in G.nodes[test_node]:
                self.detect_communities_louvain()

        net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
        net.set_options("""
        var options = {
          "nodes": { "scaling": { "min": 5, "max": 80 } },
          "edges": { "smooth": { "type": "continuous" } },
          "physics": { "barnesHut": { "gravitationalConstant": -8000, "springLength": 150 } }
        }
        """)

        # Scale node sizes
        sizes = {}
        if node_size_attr:
            raw_vals = [G.nodes[n].get(node_size_attr, 0.0) for n in G.nodes()]
            if raw_vals:
                smin, smax = min(raw_vals), max(raw_vals)
                for n in G.nodes():
                    v = G.nodes[n].get(node_size_attr, 0.0)
                    if smax == smin:
                        sizes[n] = 10
                    else:
                        sizes[n] = 5 + 75 * (v - smin) / (smax - smin)

        # Color mapping
        type_colors = {
            "channel": "#e74c3c",
            "video": "#3498db",
            "user": "#2ecc71",
            "comment": "#f39c12",
            "topic": "#9b59b6",
        }
        community_colors = [
            "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
            "#1abc9c", "#e67e22", "#c0392b", "#8e44ad", "#27ae60",
            "#d35400", "#2980b9", "#16a085", "#f1c40f", "#7f8c8d",
        ]

        for node, attrs in G.nodes(data=True):
            ntype = attrs.get("type", "unknown")
            title_parts = [f"{ntype}: {node}"]
            for k, v in attrs.items():
                if k not in ("type", "community_id"):
                    try:
                        title_parts.append(f"{k}: {v}")
                    except Exception:
                        pass
            title = "\n".join(title_parts[:10])

            if color_by == "community":
                cid = attrs.get("community_id", -1)
                color = community_colors[cid % len(community_colors)] if cid >= 0 else "#95a5a6"
            elif color_by in attrs:
                color = type_colors.get(attrs[color_by], "#95a5a6")
            else:
                color = type_colors.get(ntype, "#95a5a6")

            size = sizes.get(node, 10)
            net.add_node(node, label=str(node)[:30], title=title, color=color, size=size)

        for u, v, edata in G.edges(data=True):
            weight = edata.get("weight", 1.0)
            relation = edata.get("relation", "")
            title = f"relation: {relation}\nweight: {weight:.4f}"
            net.add_edge(u, v, title=title, value=float(weight))

        path = os.path.join(CONFIG["GRAPHS_DIR"], output_file)
        os.makedirs(CONFIG["GRAPHS_DIR"], exist_ok=True)
        net.show(path, notebook=False)
        print(f"PyVis graph exported to {path}")
        return path

    def export_to_gml(self, filename: str = "graph.gml") -> str:
        """Export graph to GML format.

        Returns
        -------
        str
            Path to the exported file.
        """
        G = self.graph
        path = os.path.join(CONFIG["GRAPHS_DIR"], filename)
        os.makedirs(CONFIG["GRAPHS_DIR"], exist_ok=True)

        # Convert non-serializable attributes to strings
        G_export = G.copy()
        for node, attrs in G_export.nodes(data=True):
            for k, v in list(attrs.items()):
                if isinstance(v, (np.ndarray, np.generic)):
                    attrs[k] = str(v)
                elif isinstance(v, (dict, list)):
                    attrs[k] = json.dumps(v)

        nx.write_gml(G_export, path)
        print(f"Graph exported to GML: {path}")
        return path

    def export_graph_stats(self, filename: str = "graph_stats.json") -> str:
        """Export graph statistics as JSON.

        Returns
        -------
        str
            Path to the exported file.
        """
        stats = self.get_graph_stats()
        path = os.path.join(CONFIG["GRAPHS_DIR"], filename)
        os.makedirs(CONFIG["GRAPHS_DIR"], exist_ok=True)

        # Add degree distribution histogram
        degrees = [d for _, d in self.graph.degree()]
        stats["degree_min"] = int(np.min(degrees)) if degrees else 0
        stats["degree_max"] = int(np.max(degrees)) if degrees else 0
        stats["degree_std"] = float(np.std(degrees)) if degrees else 0.0

        with open(path, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"Graph stats exported to {path}")
        return path

    # ==================================================================
    # 10. FULL ANALYSIS PIPELINE
    # ==================================================================

    def run_full_analysis(
        self,
        channels_df: pd.DataFrame,
        videos_df: pd.DataFrame,
        comments_df: pd.DataFrame,
        users_df: Optional[pd.DataFrame] = None,
        topics_df: Optional[pd.DataFrame] = None,
        embeddings: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Orchestrate the full graph analysis pipeline.

        Steps:
        1. Build graph
        2. Compute centrality metrics
        3. Detect communities
        4. Get community stats / themes
        5. Rank link predictions
        6. Aggregate features
        7. Export graph stats, GML
        8. Return comprehensive results dict
        """
        results: Dict[str, Any] = {}

        print("=" * 60)
        print("Starting full graph analysis pipeline")
        print("=" * 60)

        # 1. Build graph
        print("\n[1/8] Building graph...")
        self.build_graph(channels_df, videos_df, comments_df, users_df, topics_df)
        if embeddings is not None:
            self.add_video_similarity_edges(embeddings)
        results["graph_stats"] = self.get_graph_stats()
        print(f"  Nodes: {results['graph_stats']['num_nodes']}, Edges: {results['graph_stats']['num_edges']}")

        # 2. Centrality
        print("\n[2/8] Computing centrality metrics...")
        self.compute_centrality_metrics()
        results["top_videos"] = self.get_influential_videos(n=20)
        results["top_users"] = self.get_influential_users(n=20)
        results["top_topics"] = self.get_influential_topics(n=10)

        # 3. Communities
        print("\n[3/8] Detecting communities...")
        self.detect_communities_louvain()
        results["community_stats"] = self.get_community_stats()
        results["community_themes"] = self.identify_community_themes()

        # 4. Link prediction
        print("\n[4/8] Ranking link predictions (user -> video)...")
        try:
            results["link_predictions"] = self.rank_link_predictions(
                node_type_a="user", node_type_b="video", top_n=50
            )
        except Exception as e:
            logger.warning("Link prediction failed: %s", e)
            results["link_predictions"] = pd.DataFrame()

        # 5. Feature aggregation
        print("\n[5/8] Aggregating node features...")
        results["node_embeddings"] = self.aggregate_node_features()

        # 6. Simplified classifier (if success_label exists)
        print("\n[6/8] Running simplified graph classifier...")
        try:
            results["classifier"] = self.simplified_graph_classifier(
                target_node_type="video", target_attr="success_label"
            )
        except Exception as e:
            logger.warning("Simplified classifier failed: %s", e)
            results["classifier"] = {}

        # 7. Explain top nodes
        print("\n[7/8] Generating explanations for top nodes...")
        explanations = []
        if not results["top_videos"].empty:
            for node in results["top_videos"]["node"].head(3):
                explanations.append(self.explain_central_node(node))
        results["explanations"] = explanations

        # 8. Export
        print("\n[8/8] Exporting graph artifacts...")
        results["export_stats_json"] = self.export_graph_stats()
        results["export_gml"] = self.export_to_gml()
        try:
            results["export_pyvis"] = self.export_to_pyvis()
        except ImportError:
            logger.warning("PyVis export skipped (pyvis not installed).")
            results["export_pyvis"] = "skipped"

        print("\n" + "=" * 60)
        print("Full graph analysis pipeline complete.")
        print("=" * 60)

        return results


# ======================================================================
# __main__
# ======================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the graph analysis pipeline.")
    parser.add_argument(
        "--channels",
        default=os.path.join(CONFIG["PROCESSED_DATA_DIR"], "channels_processed.parquet"),
        help="Path to channels parquet.",
    )
    parser.add_argument(
        "--videos",
        default=os.path.join(CONFIG["PROCESSED_DATA_DIR"], "videos_processed.parquet"),
        help="Path to videos parquet.",
    )
    parser.add_argument(
        "--comments",
        default=os.path.join(CONFIG["PROCESSED_DATA_DIR"], "comments_processed.parquet"),
        help="Path to comments parquet.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=10000,
        help="Cap on total graph nodes (subsample comments if exceeded).",
    )
    parser.add_argument(
        "--output-dir",
        default=CONFIG["GRAPHS_DIR"],
        help="Directory to save outputs.",
    )
    args = parser.parse_args()

    # Load data
    data: Dict[str, pd.DataFrame] = {}
    for name, path in [("channels", args.channels), ("videos", args.videos), ("comments", args.comments)]:
        try:
            data[name] = pd.read_parquet(path)
            print(f"Loaded {name}: {data[name].shape} from {path}")
        except FileNotFoundError:
            print(f"File not found: {path} -- generating minimal demo {name}.")
            rng = np.random.default_rng(42)
            if name == "channels":
                data[name] = pd.DataFrame({
                    "channel_id": ["UC_demo_01", "UC_demo_02", "UC_demo_03"],
                    "channel_name": ["Demo News", "Demo Entertainment", "Demo Music"],
                    "subscriber_count": rng.integers(1000, 1000000, 3),
                    "video_count": [200, 300, 150],
                    "view_count": rng.integers(100000, 50000000, 3),
                })
            elif name == "videos":
                n_vid = 60
                ch_ids = ["UC_demo_01", "UC_demo_02", "UC_demo_03"]
                data[name] = pd.DataFrame({
                    "video_id": [f"vid_{i:04d}" for i in range(n_vid)],
                    "channel_id": rng.choice(ch_ids, n_vid),
                    "title": [f"Demo Video {i}" for i in range(n_vid)],
                    "view_count": rng.lognormal(9, 1.5, n_vid).astype(int),
                    "like_count": rng.lognormal(6, 1.8, n_vid).astype(int),
                    "comment_count": rng.lognormal(4, 2, n_vid).astype(int),
                    "engagement_rate": np.abs(rng.normal(0.04, 0.02, n_vid)),
                    "published_at": pd.to_datetime(rng.integers(
                        pd.Timestamp("2024-01-01").value // 10**9,
                        pd.Timestamp("2025-06-01").value // 10**9,
                        n_vid,
                    ), unit="s"),
                    "success_label": rng.choice(["low", "medium", "high", "viral"], n_vid),
                })
            elif name == "comments":
                n_cmt = 300
                vid_ids = [f"vid_{i:04d}" for i in range(60)]
                authors = [f"user_{i:03d}" for i in range(50)]
                data[name] = pd.DataFrame({
                    "comment_id": [f"cmt_{i:04d}" for i in range(n_cmt)],
                    "video_id": rng.choice(vid_ids, n_cmt),
                    "author_name": rng.choice(authors, n_cmt),
                    "comment_text": [f"This is demo comment number {i}" for i in range(n_cmt)],
                    "like_count": rng.poisson(3, n_cmt),
                    "sentiment_score": rng.uniform(-1, 1, n_cmt),
                })
        except Exception as exc:
            print(f"Error loading {path}: {exc} -- skipping {name}.")
            data[name] = pd.DataFrame()

    # Subsample comments to keep graph size manageable
    comments_df = data["comments"]
    n_comments = len(comments_df)
    n_videos = len(data["videos"])
    max_comments = args.max_nodes - n_videos - len(data["channels"]) - 200
    if n_comments > max_comments and max_comments > 0:
        print(f"Subsampling comments: {n_comments} -> {max_comments}")
        comments_df = comments_df.sample(n=max_comments, random_state=42).reset_index(drop=True)

    # Ensure output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run pipeline
    analyzer = GraphAnalyzer(random_state=CONFIG["RANDOM_STATE"])
    results = analyzer.run_full_analysis(
        data["channels"],
        data["videos"],
        comments_df,
        embeddings=None,
    )

    # Save results
    print("\nSaving results...")
    for key, value in results.items():
        try:
            out_path = os.path.join(args.output_dir, f"{key}.parquet")
            if isinstance(value, pd.DataFrame) and not value.empty:
                value.to_parquet(out_path, index=False)
                print(f"  Saved {key} -> {out_path}")
            elif isinstance(value, dict) and value:
                json_path = os.path.join(args.output_dir, f"{key}.json")
                try:
                    with open(json_path, "w") as f:
                        json.dump(value, f, indent=2, default=str)
                    print(f"  Saved {key} -> {json_path}")
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("Failed to save %s: %s", key, exc)

    print(f"\nGraph analysis complete. Outputs saved to {args.output_dir}")
