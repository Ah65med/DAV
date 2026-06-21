#!/usr/bin/env python3
"""End-to-end pipeline orchestrator for Digital Media Analytics project.

Runs all phases sequentially:
  1. Data Collection (real API or synthetic fallback)
  2. Preprocessing
  3. Synthetic Data Generation
  4. NLP Processing
  5. Feature Engineering
  6. Predictive Modeling
  7. Graph Analysis
  8. Visualization
  9. Decision Support

Usage:
    python src/main.py              # run full pipeline
    python src/main.py --phase 1    # run only data collection
    python src/main.py --phase 5    # run only feature engineering
    python src/main.py --skip-nlp   # skip heavy NLP steps
"""

import os
import sys
import argparse
import time
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG = {
    "CHANNELS": [
        {"name": "Aaj TV (Aaj News)", "channel_id": "UCgBAPAcLsh_MAPvJprIz89w"},
        {"name": "Hum TV", "channel_id": "UCEeEQxm6qc_qaTE7qTV5aLQ"},
        {"name": "Raftar", "channel_id": "UC6zIImBjDqtEsVZfQLPoQSw"},
    ],
    "YOUTUBE_API_KEY": os.environ.get("YOUTUBE_API_KEY", "YOUR_YOUTUBE_API_KEY_HERE"),
    "DATA_DIR": "./data",
    "RAW_DATA_DIR": "./data/raw",
    "PROCESSED_DATA_DIR": "./data/processed",
    "SYNTHETIC_DATA_DIR": "./data/synthetic",
    "OUTPUT_DIR": "./outputs",
    "FIGURES_DIR": "./outputs/figures",
    "GRAPHS_DIR": "./outputs/graphs",
    "MODELS_DIR": "./outputs/models",
    "REPORTS_DIR": "./outputs/reports",
    "RANDOM_STATE": 42,
    "BATCH_SIZE": 512,
    "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
}

for d in [
    CONFIG["DATA_DIR"],
    CONFIG["RAW_DATA_DIR"],
    CONFIG["PROCESSED_DATA_DIR"],
    CONFIG["SYNTHETIC_DATA_DIR"],
    CONFIG["OUTPUT_DIR"],
    CONFIG["FIGURES_DIR"],
    CONFIG["GRAPHS_DIR"],
    CONFIG["MODELS_DIR"],
    CONFIG["REPORTS_DIR"],
]:
    os.makedirs(d, exist_ok=True)


def phase_header(n, title):
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  PHASE {n}: {title}")
    print(f"{bar}\n")


def run_phase_1():
    """Data Collection."""
    phase_header(1, "Data Collection")
    from src.data_collection import SyntheticDataCollector

    collector = SyntheticDataCollector()
    channels_df, videos_df, comments_df, replies_df, ts_df = collector.collect_all()

    collector.save_parquet(channels_df, "channels.parquet")
    collector.save_parquet(videos_df, "videos.parquet")
    collector.save_parquet(comments_df, "comments.parquet")
    collector.save_parquet(replies_df, "comment_replies.parquet")
    collector.save_parquet(ts_df, "video_statistics_timeseries.parquet")

    print(f"  Channels: {len(channels_df)}")
    print(f"  Videos:   {len(videos_df)}")
    print(f"  Comments: {len(comments_df)}")
    print(f"  Replies:  {len(replies_df)}")
    print(f"  Time series snapshots: {len(ts_df)}")
    return channels_df, videos_df, comments_df, replies_df, ts_df


def run_phase_2():
    """Preprocessing."""
    phase_header(2, "Preprocessing")
    import pandas as pd
    from src.preprocessing import AdvancedPreprocessing

    channels_df = pd.read_parquet(f"{CONFIG['PROCESSED_DATA_DIR']}/channels.parquet")
    videos_df = pd.read_parquet(f"{CONFIG['PROCESSED_DATA_DIR']}/videos.parquet")
    comments_df = pd.read_parquet(f"{CONFIG['PROCESSED_DATA_DIR']}/comments.parquet")
    replies_df = pd.read_parquet(f"{CONFIG['PROCESSED_DATA_DIR']}/comment_replies.parquet")

    preprocessor = AdvancedPreprocessing()
    result = preprocessor.run_full_pipeline(channels_df, videos_df, comments_df, replies_df)

    v_proc = result["videos_processed"]
    c_proc = result["comments_processed"]

    v_proc.to_parquet(f"{CONFIG['PROCESSED_DATA_DIR']}/videos_processed.parquet", index=False)
    c_proc.to_parquet(f"{CONFIG['PROCESSED_DATA_DIR']}/comments_processed.parquet", index=False)

    print(f"  Processed videos shape:   {v_proc.shape}")
    print(f"  Processed comments shape: {c_proc.shape}")
    if "validation" in result:
        print(f"  Validation: {result['validation']}")
    return result


def run_phase_3():
    """Synthetic Data Generation."""
    phase_header(3, "Synthetic Data Generation")
    import pandas as pd
    from src.synthetic_generation import SyntheticDataGenerator

    videos_df = pd.read_parquet(f"{CONFIG['PROCESSED_DATA_DIR']}/videos_processed.parquet")
    comments_df = pd.read_parquet(f"{CONFIG['PROCESSED_DATA_DIR']}/comments_processed.parquet")

    generator = SyntheticDataGenerator()
    syn_results = generator.generate_all(videos_df, comments_df)

    print(f"  Synthetic videos generated:   {syn_results.get('n_synthetic_videos', 'N/A')}")
    print(f"  Augmented comments generated: {syn_results.get('n_augmented_comments', 'N/A')}")
    print(f"  Time series snapshots:        {syn_results.get('n_snapshots', 'N/A')}")
    return syn_results


def run_phase_4():
    """NLP Processing."""
    phase_header(4, "NLP Processing")
    import pandas as pd
    from src.nlp_pipeline import NLPPipeline

    videos_df = pd.read_parquet(f"{CONFIG['PROCESSED_DATA_DIR']}/videos_processed.parquet")
    try:
        comments_df = pd.read_parquet(f"{CONFIG['PROCESSED_DATA_DIR']}/comments_processed.parquet")
    except Exception:
        comments_df = pd.read_parquet(f"{CONFIG['PROCESSED_DATA_DIR']}/comments.parquet")

    nlp = NLPPipeline(batch_size=CONFIG["BATCH_SIZE"])
    results = nlp.run_full_pipeline(videos_df, comments_df)

    print(f"  Sentiment analyzed: {results.get('n_sentiment_analyzed', 'N/A')} comments")
    print(f"  Embeddings generated: {results.get('embeddings_shape', 'N/A')}")
    if "topic_distribution" in results:
        print(f"  Topics discovered: {len(results['topic_distribution'])}")
    return results


def run_phase_5():
    """Feature Engineering."""
    phase_header(5, "Feature Engineering")
    import pandas as pd
    import pickle
    from src.feature_engineering import FeatureEngineer

    videos_df = pd.read_parquet(f"{CONFIG['PROCESSED_DATA_DIR']}/videos_processed.parquet")
    comments_df = pd.read_parquet(f"{CONFIG['PROCESSED_DATA_DIR']}/comments_processed.parquet")

    try:
        with open(f"{CONFIG['OUTPUT_DIR']}/nlp_results.pkl", "rb") as f:
            nlp_results = pickle.load(f)
    except Exception:
        nlp_results = {}

    try:
        with open(f"{CONFIG['OUTPUT_DIR']}/embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)
    except Exception:
        embeddings = {}

    engineer = FeatureEngineer()
    feature_matrix, metadata = engineer.build_feature_matrix(
        videos_df, comments_df, nlp_results, embeddings
    )

    engineer.save_features(feature_matrix, "feature_matrix.parquet")
    print(f"  Feature matrix shape: {feature_matrix.shape}")
    print(f"  Feature groups: {list(metadata.keys())}")
    return feature_matrix, metadata


def run_phase_6():
    """Predictive Modeling."""
    phase_header(6, "Predictive Modeling")
    import pandas as pd
    from src.modeling import ModelingPipeline

    feature_matrix = pd.read_parquet(f"{CONFIG['PROCESSED_DATA_DIR']}/feature_matrix.parquet")

    pipeline = ModelingPipeline()
    results = pipeline.run_full_modeling_pipeline(feature_matrix)

    print("\n  --- Regression Results ---")
    for name, metrics in results.get("regression", {}).items():
        if isinstance(metrics, dict) and "R2" in metrics:
            print(f"  {name}: R²={metrics['R2']:.3f}, MAE={metrics['MAE']:.3f}, RMSE={metrics['RMSE']:.3f}")

    print("\n  --- Classification Results ---")
    for name, metrics in results.get("classification", {}).items():
        if isinstance(metrics, dict) and "accuracy" in metrics:
            print(f"  {name}: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1_weighted']:.3f}")

    print("\n  --- Clustering Results ---")
    for name, metrics in results.get("clustering", {}).items():
        if isinstance(metrics, dict) and "silhouette" in metrics:
            print(f"  {name}: Silhouette={metrics['silhouette']:.3f}")
    return results


def run_phase_7():
    """Graph Analysis."""
    phase_header(7, "Graph Analysis")
    import pandas as pd
    from src.graph_analysis import GraphAnalyzer

    channels_df = pd.read_parquet(f"{CONFIG['PROCESSED_DATA_DIR']}/channels.parquet")
    videos_df = pd.read_parquet(f"{CONFIG['PROCESSED_DATA_DIR']}/videos_processed.parquet")
    comments_df = pd.read_parquet(f"{CONFIG['PROCESSED_DATA_DIR']}/comments_processed.parquet")

    analyzer = GraphAnalyzer()
    result = analyzer.run_full_analysis(channels_df, videos_df, comments_df)

    print(f"  Graph nodes: {result.get('n_nodes', 'N/A')}")
    print(f"  Graph edges: {result.get('n_edges', 'N/A')}")
    print(f"  Communities: {result.get('n_communities', 'N/A')}")
    print(f"  Top video by PageRank: {result.get('top_video', 'N/A')}")
    return result


def run_phase_8():
    """Visualization."""
    phase_header(8, "Visualization")
    import pandas as pd
    from src.visualization import VisualizationEngine

    videos_df = pd.read_parquet(f"{CONFIG['PROCESSED_DATA_DIR']}/videos_processed.parquet")
    comments_df = pd.read_parquet(f"{CONFIG['PROCESSED_DATA_DIR']}/comments_processed.parquet")

    viz = VisualizationEngine()
    viz.generate_all_visualizations(videos_df, comments_df)

    print("  All visualizations generated and saved to outputs/figures/")
    print("  Interactive graphs saved to outputs/graphs/")
    print("  Streamlit dashboard script at outputs/streamlit_dashboard.py")


def run_phase_9():
    """Decision Support."""
    phase_header(9, "Decision Support Framework")
    import pandas as pd
    import pickle
    from src.decision_support import DecisionSupportEngine

    videos_df = pd.read_parquet(f"{CONFIG['PROCESSED_DATA_DIR']}/videos_processed.parquet")
    comments_df = pd.read_parquet(f"{CONFIG['PROCESSED_DATA_DIR']}/comments_processed.parquet")

    try:
        with open(f"{CONFIG['OUTPUT_DIR']}/modeling_results.pkl", "rb") as f:
            modeling_results = pickle.load(f)
    except Exception:
        modeling_results = {}

    try:
        with open(f"{CONFIG['OUTPUT_DIR']}/graph_results.pkl", "rb") as f:
            graph_results = pickle.load(f)
    except Exception:
        graph_results = {}

    try:
        with open(f"{CONFIG['OUTPUT_DIR']}/nlp_results.pkl", "rb") as f:
            nlp_results = pickle.load(f)
    except Exception:
        nlp_results = {}

    engine = DecisionSupportEngine()
    recommendations = engine.run_full_decision_support(
        videos_df, comments_df, modeling_results, graph_results, nlp_results
    )

    print(f"  Generated {len(recommendations)} recommendations")

    for i, rec in enumerate(recommendations[:5], 1):
        print(f"\n  RECOMMENDATION #{i}: {rec.recommendation}")
        print(f"  CONFIDENCE: {rec.confidence}")
    return recommendations


def main():
    parser = argparse.ArgumentParser(description="Digital Media Analytics Pipeline")
    parser.add_argument("--phase", type=int, choices=range(1, 10), help="Run a specific phase (1-9)")
    parser.add_argument("--skip-nlp", action="store_true", help="Skip NLP processing (phase 4)")
    parser.add_argument("--skip-graph", action="store_true", help="Skip graph analysis (phase 7)")
    parser.add_argument("--start-from", type=int, choices=range(1, 10), help="Start from a specific phase")
    args = parser.parse_args()

    total_start = time.time()

    print("=" * 70)
    print("  DIGITAL MEDIA ANALYTICS")
    print("  with Intelligent Decision Support System")
    print("=" * 70)
    print(f"  Data directory:  {CONFIG['DATA_DIR']}")
    print(f"  Output directory: {CONFIG['OUTPUT_DIR']}")
    print(f"  Random state:    {CONFIG['RANDOM_STATE']}")
    print(f"  API key set:     {'Yes' if CONFIG['YOUTUBE_API_KEY'] not in ('', 'YOUR_YOUTUBE_API_KEY_HERE') else 'No (using synthetic data)'}")
    print("=" * 70)

    start_phase = args.start_from or args.phase or 1
    run_single = args.phase is not None

    results = {}

    if start_phase <= 1 and (not run_single or args.phase == 1):
        results["data"] = run_phase_1()

    if start_phase <= 2 and (not run_single or args.phase == 2):
        results["preprocessing"] = run_phase_2()

    if start_phase <= 3 and (not run_single or args.phase == 3):
        results["synthetic"] = run_phase_3()

    if start_phase <= 4 and not args.skip_nlp and (not run_single or args.phase == 4):
        try:
            results["nlp"] = run_phase_4()
        except Exception as e:
            print(f"  [WARNING] NLP phase failed: {e}")
            print("  Continuing without NLP results...")

    if start_phase <= 5 and (not run_single or args.phase == 5):
        results["features"] = run_phase_5()

    if start_phase <= 6 and (not run_single or args.phase == 6):
        results["modeling"] = run_phase_6()

    if start_phase <= 7 and not args.skip_graph and (not run_single or args.phase == 7):
        try:
            results["graph"] = run_phase_7()
        except Exception as e:
            print(f"  [WARNING] Graph analysis failed: {e}")
            print("  Continuing without graph results...")

    if start_phase <= 8 and (not run_single or args.phase == 8):
        results["visualization"] = run_phase_8()

    if start_phase <= 9 and (not run_single or args.phase == 9):
        results["decision_support"] = run_phase_9()

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"  PIPELINE COMPLETE — Total time: {total_elapsed:.1f}s")
    print("=" * 70)
    print(f"  Outputs saved to: {CONFIG['OUTPUT_DIR']}/")
    print(f"  Processed data:   {CONFIG['PROCESSED_DATA_DIR']}/")
    print(f"  Figures:          {CONFIG['FIGURES_DIR']}/")
    print(f"  Reports:          {CONFIG['REPORTS_DIR']}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
