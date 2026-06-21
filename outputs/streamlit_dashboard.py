"""
Q13: Visualization & Interpretability Dashboard
Digital Media Analytics — Pakistani YouTube Channels
Run: streamlit run streamlit_dashboard.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Digital Media Analytics",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif; }
.stApp { background: #0A0E1A; color: #E8EAF0; }
section[data-testid="stSidebar"] { background: #0F1525; border-right: 1px solid #1E2640; }
section[data-testid="stSidebar"] * { color: #C8CCDC !important; }
.metric-card {
    background: linear-gradient(135deg, #12192E 0%, #1A2340 100%);
    border: 1px solid #1E2D50; border-radius: 12px;
    padding: 16px 18px; margin: 4px 0; min-height: 90px;
}
.metric-value {
    font-family: 'Syne', sans-serif; font-size: 26px; font-weight: 800;
    color: #4D9FFF; margin: 0; line-height: 1.2;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.metric-label { font-size: 11px; color: #6B7A99; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 6px; }
.metric-delta { font-size: 12px; color: #3DD68C; margin-top: 4px; }
.section-header {
    font-family: 'Syne', sans-serif; font-size: 20px; font-weight: 700;
    color: #E8EAF0; border-left: 3px solid #4D9FFF;
    padding-left: 12px; margin: 24px 0 16px;
}
.insight-box {
    background: #12192E; border: 1px solid #1E2D50;
    border-left: 3px solid #4D9FFF; border-radius: 8px;
    padding: 14px 18px; margin: 8px 0; font-size: 13px;
    color: #A8B0C8; line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

CH_COLORS = {
    'Aaj TV Official'  : '#4D9FFF',
    'HUM TV'           : '#3DD68C',
    'Geo Entertainment': '#FF8C42',
    'Raftar'           : '#C47FFF',
}

def L(**kwargs):
    base = dict(
        paper_bgcolor='#12192E',
        plot_bgcolor ='#12192E',
        font         =dict(color='#A8B0C8', family='DM Sans'),
        legend       =dict(bgcolor='#1A2340', bordercolor='#1E2D50', borderwidth=1),
        margin       =dict(l=40, r=20, t=50, b=40),
        colorway     =['#4D9FFF','#3DD68C','#FF8C42','#C47FFF','#FF6B9D','#FFD93D'],
        title_font   =dict(color='#E8EAF0', family='Syne', size=14),
        xaxis        =dict(gridcolor='#1E2D50', linecolor='#1E2D50',
                           zerolinecolor='#1E2D50', tickfont=dict(size=11)),
        yaxis        =dict(gridcolor='#1E2D50', linecolor='#1E2D50',
                           zerolinecolor='#1E2D50', tickfont=dict(size=11)),
    )
    base.update(kwargs)
    return base

BASE = r'C:/Users/PC/DAV/DAV/data'
FIGURES_DIR = r'C:/Users/PC/DAV/DAV/outputs/figures'

def fig_img(name, caption=None, width=None):
    """Display a saved figure from outputs/figures/, or skip silently if missing."""
    path = os.path.join(FIGURES_DIR, name)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=(width is None), width=width)
    else:
        st.caption(f"_(figure `{name}` not yet generated — run notebook first)_")

@st.cache_data
def load_data():
    d = {}
    files = {
        'videos'    : f'{BASE}/processed/videos_processed.parquet',
        'combined'  : f'{BASE}/processed/combined_videos.parquet',
        'comments'  : f'{BASE}/processed/comments_processed.parquet',
        'nlp'       : f'{BASE}/processed/comments_nlp_enriched.parquet',
        'fm'        : f'{BASE}/processed/feature_matrix.parquet',
        'fm_comb'   : f'{BASE}/processed/feature_matrix_combined.parquet',
        'syn_videos': f'{BASE}/synthetic/synthetic_videos_large.parquet',
        'timeseries': f'{BASE}/synthetic/synthetic_timeseries_large.parquet',
    }
    for k, path in files.items():
        try:
            d[k] = pd.read_parquet(path)
        except Exception:
            d[k] = pd.DataFrame()
    name_map = {
        'Aaj TV (Aaj News)': 'Aaj TV Official',
        'Hum TV'           : 'HUM TV',
        'Har Pal Geo'      : 'Geo Entertainment',
        'Raftar'           : 'Raftar',
    }
    for k in ['videos', 'combined', 'syn_videos']:
        if not d[k].empty and 'channel_name' in d[k].columns:
            d[k]['channel_name'] = d[k]['channel_name'].map(
                name_map).fillna(d[k]['channel_name'])
    for k in ['videos', 'combined']:
        if not d[k].empty and 'published_at' in d[k].columns:
            d[k]['published_at'] = pd.to_datetime(d[k]['published_at'], errors='coerce')
            d[k]['hour']         = d[k]['published_at'].dt.hour
            d[k]['day_of_week']  = d[k]['published_at'].dt.dayofweek
    return d

with st.spinner("Loading data..."):
    D = load_data()

videos     = D['videos']
combined   = D['combined']
comments   = D['comments']
nlp        = D['nlp']
fm         = D['fm']
fm_comb    = D['fm_comb']
syn_videos = D['syn_videos']
timeseries = D['timeseries']

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📺 DAV Dashboard")
    st.markdown("---")
    page = st.radio("Navigation", [
        "📊 Overview",
        "🌐 Network Analysis",
        "📈 Temporal Analysis",
        "💬 NLP & Sentiment",
        "🤖 ML Interpretability",
        "🧠 XGNN Explainability",
        "🎯 Decision Support",
    ])
    st.markdown("---")
    st.markdown("**Filters**")
    all_channels = ['All'] + sorted(videos['channel_name'].dropna().unique().tolist()) \
        if not videos.empty else ['All']
    sel_channel = st.selectbox("Channel", all_channels)
    data_source = st.radio("Data source", ["Real only", "Real + Synthetic"])
    st.markdown("---")
    real_n   = len(videos)
    syn_n    = len(syn_videos)
    comb_n   = len(combined)
    comm_n   = len(comments)
    nlp_n    = len(nlp)
    st.markdown(f"""
    <div style='font-size:11px;color:#4B5270;line-height:1.8;'>
    Real videos: {real_n:,}<br>
    Synthetic: {syn_n:,}<br>
    Combined: {comb_n:,}<br>
    Comments: {comm_n:,}<br>
    NLP comments: {nlp_n:,}
    </div>""", unsafe_allow_html=True)

# Active dataset respects data_source toggle
def active_df():
    """Return real or combined df based on sidebar toggle."""
    df = combined if data_source == "Real + Synthetic" else videos
    return df

def fdf(df, col='channel_name'):
    if sel_channel != 'All' and col in df.columns:
        return df[df[col] == sel_channel].copy()
    return df.copy()

# ══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("# Digital Media Analytics")
    st.markdown("##### Pakistani YouTube Channels — Aaj TV · HUM TV · Geo · Raftar")
    if data_source == "Real + Synthetic":
        st.info("Showing **Real + Synthetic** data. Toggle sidebar to Real only for raw channel stats.")
    st.markdown("---")

    df = fdf(active_df())

    top_ch = df.groupby('channel_name')['views'].sum().idxmax() \
        if not df.empty and 'channel_name' in df.columns and 'views' in df.columns else 'N/A'

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, label, delta in [
        (c1, f"{df['views'].sum()/1e6:.1f}M",          "Total Views",      f"{'Real+Syn' if data_source=='Real + Synthetic' else 'Real only'}"),
        (c2, f"{df['engagement_rate'].mean():.4f}",     "Avg Engagement",   ""),
        (c3, f"{len(df):,}",                            "Videos",           ""),
        (c4, f"{df['likes'].mean():,.0f}",              "Avg Likes",        ""),
        (c5, str(top_ch)[:14],                          "Top Channel",      "by views"),
    ]:
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{label}</div>
            <div class='metric-delta'>{delta}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Channel Performance</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        cs = fdf(videos).groupby('channel_name').agg(
            Views=('views', 'sum'),
            Videos=('video_id', 'count'),
            Engagement=('engagement_rate', 'mean')
        ).reset_index()
        fig = px.bar(cs.sort_values('Views', ascending=True),
                     x='Views', y='channel_name', orientation='h',
                     color='channel_name', color_discrete_map=CH_COLORS,
                     title='Total Views by Channel (Real)')
        fig.update_traces(showlegend=False)
        fig.update_layout(**L())
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.scatter(
            fdf(videos).dropna(subset=['views', 'engagement_rate']),
            x='views', y='engagement_rate',
            color='channel_name', color_discrete_map=CH_COLORS,
            size='likes', size_max=25, hover_data=['title'],
            title='Views vs Engagement Rate', log_x=True)
        fig2.update_layout(**L())
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.box(fdf(videos), x='channel_name', y='engagement_rate',
                      color='channel_name', color_discrete_map=CH_COLORS,
                      title='Engagement Rate Distribution')
        fig3.update_traces(showlegend=False)
        fig3.update_layout(**L())
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        rc  = videos.groupby('channel_name').size().reset_index(name='Real')
        sc2 = syn_videos.groupby('channel_name').size().reset_index(name='Synthetic') \
            if not syn_videos.empty else pd.DataFrame(columns=['channel_name', 'Synthetic'])
        comp = rc.merge(sc2, on='channel_name', how='outer').fillna(0)
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(name='Real',      x=comp['channel_name'],
                              y=comp['Real'],      marker_color='#4D9FFF'))
        fig4.add_trace(go.Bar(name='Synthetic', x=comp['channel_name'],
                              y=comp['Synthetic'], marker_color='#FF8C42', opacity=0.6))
        fig4.update_layout(**L(barmode='group', title='Real vs Synthetic Videos per Channel'))
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("<div class='section-header'>Saved Figures</div>", unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        fig_img('engagement_comparison.png', 'Engagement Comparison')
    with col6:
        fig_img('correlation_heatmap.png', 'Correlation Heatmap')

    st.markdown("<div class='section-header'>Top 10 Videos</div>", unsafe_allow_html=True)
    t10 = fdf(videos).nlargest(10, 'views')[
        ['title', 'channel_name', 'views', 'likes', 'engagement_rate']
    ].reset_index(drop=True)
    t10.index += 1
    st.dataframe(
        t10.style.format({'views': '{:,.0f}', 'likes': '{:,.0f}', 'engagement_rate': '{:.4f}'}),
        use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — NETWORK ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "🌐 Network Analysis":
    st.markdown("# Network Analysis")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, "4,029",   "Total Nodes"),
        (c2, "6,179",   "Total Edges"),
        (c3, "0.00145", "Graph Density"),
        (c4, "97",      "Communities (Louvain)"),
    ]:
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Graph Topology</div>", unsafe_allow_html=True)
    fig_img('graph_topology_sample.png', 'Heterogeneous Graph Topology Sample (up to 300 nodes)')

    st.markdown("<div class='section-header'>Graph Structure</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        nd = pd.DataFrame({
            'Type':  ['Comment', 'User', 'Video', 'Channel'],
            'Count': [1000,       828,    247,      4],
        })
        fig = px.pie(nd, values='Count', names='Type', hole=0.55,
                     color='Type',
                     color_discrete_map={'Comment': '#FF8C42', 'User': '#3DD68C',
                                         'Video': '#4D9FFF', 'Channel': '#C47FFF'},
                     title='Node Type Distribution')
        fig.update_layout(**L())
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        ed = pd.DataFrame({
            'Edge Type': ['User→Comment', 'Video→Comment', 'Channel→Video', 'User→Video'],
            'Count':     [1000,            1000,             247,             886],
        })
        fig2 = px.bar(ed, x='Count', y='Edge Type', orientation='h',
                      color='Count', color_continuous_scale='Blues',
                      title='Edge Type Distribution')
        fig2.update_layout(**L())
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-header'>Centrality & Communities</div>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        pr = pd.DataFrame({
            'Video': ['Aye Dil Ep 13', 'Leader Ep 09', 'Hadd Ep 48', 'Hadd Ep 49 Teaser',
                      'Leader Ep 10 Teaser', 'Petrol Price Hike', 'Aye Dil Ep 14 Teaser',
                      'Fuel Price Hike', 'Petrol Public Reaction', 'Hadd Ep 49 Promo'],
            'PageRank': [0.049745, 0.041621, 0.033350, 0.013976, 0.013027,
                         0.005611, 0.004804, 0.003285, 0.003122, 0.003039],
            'Channel': ['HUM TV', 'HUM TV', 'HUM TV', 'HUM TV', 'HUM TV',
                        'Aaj TV Official', 'HUM TV', 'Aaj TV Official',
                        'Aaj TV Official', 'HUM TV'],
        })
        fig3 = px.bar(pr.sort_values('PageRank'), x='PageRank', y='Video', orientation='h',
                      color='Channel', color_discrete_map=CH_COLORS,
                      title='Top 10 Videos — PageRank')
        fig3.update_layout(**L(
            yaxis=dict(gridcolor='#1E2D50', linecolor='#1E2D50',
                       zerolinecolor='#1E2D50', tickfont=dict(size=9))))
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        # 97 communities — top 15 by size
        sizes = [355, 350, 337, 133, 109, 97, 88, 62, 44, 33, 18, 15, 12, 10, 8]
        cd = pd.DataFrame({'Community': [f'C{i}' for i in range(len(sizes))], 'Size': sizes})
        fig4 = px.bar(cd, x='Community', y='Size', color='Size',
                      color_continuous_scale='Blues',
                      title='Community Sizes (Top 15 of 97)')
        fig4.update_layout(**L())
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("<div class='section-header'>Link Prediction</div>", unsafe_allow_html=True)
    ld = pd.DataFrame({
        'User':  ['@_Save_Palestine_15', '@_Save_Palestine_15', '@_Save_Palestine_15',
                  '@muhammadarshadjaved', '@muhammadarshadjaved', '@Kashaf_Ahmad',
                  '@muhammadarshadjaved', '@Kashaf_Ahmad', '@HMM-Queen-246', '@AhteshamAkram-h3z'],
        'Video': ['Leader Ep 09', 'Aye Dil Ep 13', 'Hadd Ep 48', 'Leader Ep 09',
                  'Aye Dil Ep 13', 'Aye Dil Ep 13', 'Hadd Ep 48', 'Hadd Ep 48',
                  'Hadd Ep 48', 'Leader Ep 09'],
        'Score': [0.25, 0.2473, 0.2373, 0.2000, 0.1978, 0.1978, 0.1898, 0.1898, 0.1661, 0.1499],
        'PrefAttach': [3750, 3710, 3560, 3000, 2968, 2968, 2848, 2848, 2492, 2250],
    })
    fig5 = px.scatter(ld, x='PrefAttach', y='Score', color='Video',
                      size='Score', size_max=20, hover_data=['User'],
                      title='Link Prediction — Score vs Preferential Attachment')
    fig5.update_layout(**L())
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("""
    <div class='insight-box'>
    <b>Network Insight:</b> Greedy modularity detected <b>97 communities</b> — top 3 clusters
    (330–355 nodes) are HUM TV drama hubs. Graph density = 0.00145 (sparse). Jaccard/Adamic-Adar
    both 0 due to sparsity; preferential attachment drives link prediction scores.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 3 — TEMPORAL
# ══════════════════════════════════════════════════════════════
elif page == "📈 Temporal Analysis":
    st.markdown("# Temporal Analysis")
    st.markdown("---")

    df = fdf(active_df())

    st.markdown("<div class='section-header'>LSTM Forecast Results</div>", unsafe_allow_html=True)
    fig_img('lstm_results.png', 'LSTM Time-Series Prediction — Training vs Validation Loss & Forecast')

    st.markdown("<div class='section-header'>Publishing Patterns</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if 'hour' in df.columns and df['hour'].notna().any():
            he = df.groupby('hour')['engagement_rate'].mean().reset_index()
            fig = px.line(he, x='hour', y='engagement_rate',
                          title='Avg Engagement by Hour of Day', markers=True)
            fig.update_traces(line_color='#4D9FFF', line_width=2.5,
                              marker=dict(size=6, color='#FF8C42'))
            for h in [10, 12, 20]:
                fig.add_vline(x=h, line_dash='dash', line_color='#FF6B9D',
                              annotation_text=f'{h}:00', annotation_font_color='#FF6B9D')
            fig.update_layout(**L())
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Hour data not available for this dataset.")
    with col2:
        dm = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
        if 'day_of_week' in df.columns and df['day_of_week'].notna().any():
            de = df.groupby('day_of_week')['engagement_rate'].mean().reset_index()
            de['day']  = de['day_of_week'].map(dm)
            de['peak'] = de['day'].isin(['Wed', 'Tue', 'Thu'])
            fig2 = px.bar(de, x='day', y='engagement_rate', color='peak',
                          color_discrete_map={True: '#FF8C42', False: '#4D9FFF'},
                          title='Engagement by Day of Week',
                          category_orders={'day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']})
            fig2.update_traces(showlegend=False)
            fig2.update_layout(**L())
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Day-of-week data not available for this dataset.")

    st.markdown("<div class='section-header'>Time Series — View Growth</div>", unsafe_allow_html=True)
    if not timeseries.empty and 'age_days' in timeseries.columns:
        ta = timeseries.groupby('age_days').agg(
            views=('views', 'mean'),
            likes=('likes', 'mean')
        ).reset_index()
        ta = ta[ta['age_days'] <= 365]
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Scatter(x=ta['age_days'], y=ta['views'], name='Avg Views',
                                  line=dict(color='#4D9FFF', width=2.5),
                                  fill='tozeroy', fillcolor='rgba(77,159,255,0.1)'),
                       secondary_y=False)
        fig3.add_trace(go.Scatter(x=ta['age_days'], y=ta['likes'], name='Avg Likes',
                                  line=dict(color='#3DD68C', width=2)),
                       secondary_y=True)
        fig3.update_layout(**L(title='View & Like Growth Over Video Lifetime (Synthetic)'))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<div class='section-header'>Monthly Publishing Volume</div>", unsafe_allow_html=True)
    if 'published_at' in videos.columns:
        v2 = videos.copy()
        v2['pub_month'] = v2['published_at'].dt.to_period('M').astype(str)
        mo = v2.groupby(['pub_month', 'channel_name']).size().reset_index(name='count')
        fig4 = px.line(mo, x='pub_month', y='count', color='channel_name',
                       color_discrete_map=CH_COLORS,
                       title='Monthly Video Publishing Volume', markers=True)
        fig4.update_layout(**L(
            xaxis=dict(gridcolor='#1E2D50', linecolor='#1E2D50',
                       zerolinecolor='#1E2D50', tickfont=dict(size=10), tickangle=45)))
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""
    <div class='insight-box'>
    <b>Temporal Insights:</b> Peak publishing hours 10:00, 12:00, 20:00.
    Wed/Tue/Thu highest engagement. Weekend lift only +0.6%.
    Engagement decays rapidly after 30 days. LSTM trained over synthetic
    time-series shows strong view-growth trend in first 90 days.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 4 — NLP
# ══════════════════════════════════════════════════════════════
elif page == "💬 NLP & Sentiment":
    st.markdown("# NLP & Sentiment Analysis")
    st.markdown("---")

    st.markdown("<div class='section-header'>Multilingual Embeddings (MiniLM)</div>", unsafe_allow_html=True)
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        fig_img('bert_embeddings_pca.png', 'MiniLM Embeddings — PCA Projection')
    with col_e2:
        fig_img('nlp_multilingual_sentiment.png', 'Multilingual Sentiment Distribution')

    if not nlp.empty:
        sc_col = 'sentiment_label' if 'sentiment_label' in nlp.columns else \
                 'sentiment_label_vader' if 'sentiment_label_vader' in nlp.columns else None

        if sc_col:
            st.markdown("<div class='section-header'>Sentiment Breakdown</div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                rn = nlp[nlp['is_synthetic'] == 0] if 'is_synthetic' in nlp.columns else nlp
                rv = rn[sc_col].value_counts().reset_index()
                rv.columns = ['Sentiment', 'Count']
                fig = px.pie(rv, values='Count', names='Sentiment', hole=0.5,
                             color='Sentiment',
                             color_discrete_map={'positive': '#3DD68C',
                                                 'neutral': '#6B7A99',
                                                 'negative': '#FF6B9D'},
                             title=f'Real Comments (n={len(rn):,})')
                fig.update_layout(**L())
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                if 'is_synthetic' in nlp.columns:
                    sn = nlp[nlp['is_synthetic'] == 1]
                    sv = sn[sc_col].value_counts().reset_index()
                    sv.columns = ['Sentiment', 'Count']
                    fig2 = px.pie(sv, values='Count', names='Sentiment', hole=0.5,
                                  color='Sentiment',
                                  color_discrete_map={'positive': '#3DD68C',
                                                      'neutral': '#6B7A99',
                                                      'negative': '#FF6B9D'},
                                  title=f'Synthetic Comments (n={len(sn):,})')
                    fig2.update_layout(**L())
                    st.plotly_chart(fig2, use_container_width=True)

        sc2_col = 'sentiment_score' if 'sentiment_score' in nlp.columns else \
                  'sentiment_score_vader' if 'sentiment_score_vader' in nlp.columns else None

        if sc2_col:
            st.markdown("<div class='section-header'>Score Distribution</div>", unsafe_allow_html=True)
            fig3 = go.Figure()
            if 'is_synthetic' in nlp.columns:
                r = nlp[nlp['is_synthetic'] == 0][sc2_col].dropna()
                s = nlp[nlp['is_synthetic'] == 1][sc2_col].dropna()
                fig3.add_trace(go.Histogram(x=r, name=f'Real (n={len(r):,})',
                               marker_color='#4D9FFF', opacity=0.8,
                               nbinsx=40, histnorm='density'))
                fig3.add_trace(go.Histogram(x=s, name=f'Synthetic (n={len(s):,})',
                               marker_color='#FF8C42', opacity=0.5,
                               nbinsx=40, histnorm='density'))
            else:
                r = nlp[sc2_col].dropna()
                fig3.add_trace(go.Histogram(x=r, marker_color='#4D9FFF',
                               nbinsx=40, histnorm='density'))
            fig3.add_vline(x=0, line_dash='dash', line_color='#FF6B9D',
                           annotation_text='Neutral')
            fig3.update_layout(**L(barmode='overlay',
                                   title='Sentiment Score Density — Real vs Synthetic'))
            st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<div class='section-header'>Sentiment Timeline</div>", unsafe_allow_html=True)
    fig_img('sentiment_timeline.png', 'Sentiment Score Over Time')

    st.markdown("<div class='section-header'>Topic Keywords</div>", unsafe_allow_html=True)
    topics = {
        'Topic 1 — Urdu':       'ہے, کی, اور, کا, نہیں',
        'Topic 2 — Drama':      'drama, best, favourite, nice',
        'Topic 3 — Episodes':   'week, episode, upload, ep',
        'Topic 6 — Characters': 'zain, tariq, hina, mirza',
        'Topic 7 — Hadd':       'mirha, asfand, anaiza',
        'Topic 8 — Emotional':  'love, allah, mohammad',
        'Topic 10 — News':      'pakistan, petrol, awam',
    }
    cols = st.columns(3)
    for i, (t, w) in enumerate(topics.items()):
        with cols[i % 3]:
            st.markdown(f"""
            <div class='insight-box'>
            <b style='color:#4D9FFF'>{t}</b><br>
            <span style='color:#A8B0C8'>{w}</span>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 5 — ML INTERPRETABILITY
# ══════════════════════════════════════════════════════════════
elif page == "🤖 ML Interpretability":
    st.markdown("# ML Interpretability")
    st.markdown("---")

    st.markdown("<div class='section-header'>SHAP Feature Importance</div>", unsafe_allow_html=True)
    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        fig_img('shap_summary.png', 'SHAP Summary Plot — Random Forest Classifier')
    with col_s2:
        st.markdown("""
        <div class='insight-box'>
        <b>SHAP Insights</b><br><br>
        SHAP (SHapley Additive exPlanations) uses TreeExplainer on the Random Forest classifier.<br><br>
        Features are ranked by mean |SHAP value| across the test set.
        Red = high feature value pushes prediction higher; blue = low value pulls it down.
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Feature Engineering PCA</div>", unsafe_allow_html=True)
    fig_img('feature_engineering_pca.png', 'High-Dimensional Feature Space — PCA Projection')

    st.markdown("<div class='section-header'>Clustering Results</div>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Combined", "Per Channel"])
    with tab1:
        fig_img('clustering_combined_comparison.png', 'Clustering Comparison — Combined Dataset')
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig_img('clustering_Real_HUM_TV.png', 'HUM TV Clusters')
            fig_img('clustering_Real_Geo_Entertainment.png', 'Geo Entertainment Clusters')
        with c2:
            fig_img('clustering_Real_Aaj_TV_Official.png', 'Aaj TV Official Clusters')
            fig_img('clustering_Real_Raftar.png', 'Raftar Clusters')

    st.markdown("<div class='section-header'>Clustering Quality Metrics</div>", unsafe_allow_html=True)
    cl = pd.DataFrame({
        'Channel':    ['HUM TV', 'Aaj TV Official', 'Geo Entertainment', 'Raftar'],
        'Silhouette': [0.9060,   0.7350,            0.4122,              0.3375],
        'DB Index':   [0.1725,   0.4554,            0.8126,              0.9498],
        'Videos':     [101,      102,               33,                  11],
    })
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(cl, x='Channel', y='Silhouette', color='Channel',
                     color_discrete_map=CH_COLORS, title='Silhouette Score by Channel')
        fig.add_hline(y=0.5, line_dash='dash', line_color='#FF6B9D',
                      annotation_text='Good threshold (0.5)')
        fig.update_traces(showlegend=False)
        fig.update_layout(**L())
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.scatter(cl, x='Silhouette', y='DB Index', color='Channel',
                          size='Videos', color_discrete_map=CH_COLORS,
                          text='Channel', size_max=40,
                          title='Silhouette vs Davies-Bouldin Index')
        fig2.add_hline(y=1.0, line_dash='dash', line_color='#FF6B9D',
                       annotation_text='DB target < 1.0')
        fig2.update_layout(**L())
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-header'>Feature Importance (Random Forest)</div>", unsafe_allow_html=True)
    # Try to compute from actual feature matrix, fall back to notebook output values
    fi_computed = False
    if not fm_comb.empty:
        feature_cols = [c for c in fm_comb.columns
                        if c not in ('channel_name', 'success_label', 'video_id', 'published_at')]
        if 'success_label' in fm_comb.columns and len(feature_cols) >= 3:
            try:
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.preprocessing import LabelEncoder
                _le = LabelEncoder()
                _X  = fm_comb[feature_cols].fillna(0)
                _y  = _le.fit_transform(fm_comb['success_label'].fillna('medium'))
                _rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
                _rf.fit(_X, _y)
                fi = pd.DataFrame({
                    'Feature':    feature_cols,
                    'Importance': _rf.feature_importances_,
                }).sort_values('Importance').tail(12)
                fi_computed = True
            except Exception:
                pass

    if not fi_computed:
        fi = pd.DataFrame({
            'Feature':    ['views', 'log_views', 'views_per_day', 'likes', 'log_likes',
                           'comments_per_day', 'comments', 'growth_rate',
                           'engagement_rate', 'video_age_days'],
            'Importance': [0.312, 0.198, 0.156, 0.089, 0.071,
                           0.054, 0.042, 0.038, 0.025, 0.015],
        }).sort_values('Importance')

    title_suffix = '(from data)' if fi_computed else '(notebook run values)'
    fig3 = px.bar(fi, x='Importance', y='Feature', orientation='h',
                  color='Importance', color_continuous_scale='Blues',
                  title=f'Top Feature Importances — Random Forest {title_suffix}')
    fig3.update_layout(**L())
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    <div class='insight-box'>
    <b>ML Insights:</b> HUM TV silhouette 0.906 — drama serials cluster cleanly by engagement profile.
    Global K-Means improved from silhouette −0.18 to 0.91 after log-transform + per-channel
    stratification. SHAP confirms <i>views</i> and <i>log_views</i> dominate classification decisions.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 6 — XGNN
# ══════════════════════════════════════════════════════════════
elif page == "🧠 XGNN Explainability":
    st.markdown("# XGNN — Explainable Graph Neural Network")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, "99.8%",  "Node Classification Accuracy"),
        (c2, "4,029",  "Total Nodes"),
        (c3, "63",     "Feature Dimensions"),
        (c4, "50/50",  "Own / Neighbour Split"),
    ]:
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Feature Importance</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        xf = pd.DataFrame({
            'Feature':    ['total_views', 'nbr_max_eigenvector', 'eigenvector',
                           'nbr_mean_eigenvector', 'nbr_max_total_views',
                           'pagerank', 'closeness', 'nbr_mean_closeness',
                           'like_count', 'views'],
            'Importance': [0.0741, 0.0694, 0.0681, 0.0535, 0.0422,
                           0.0204, 0.0045, 0.0020, 0.0016, 0.0012],
            'Type':       ['Own', 'Neighbour', 'Own', 'Neighbour', 'Neighbour',
                           'Own', 'Own', 'Neighbour', 'Own', 'Own'],
        }).sort_values('Importance')
        fig = px.bar(xf, x='Importance', y='Feature', orientation='h', color='Type',
                     color_discrete_map={'Own': '#4D9FFF', 'Neighbour': '#C47FFF'},
                     title='Permutation Feature Importance')
        fig.update_layout(**L())
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = go.Figure(go.Pie(
            values=[0.170, 0.168],
            labels=['Own (50.2%)', 'Neighbour (49.8%)'],
            hole=0.55,
            marker_colors=['#4D9FFF', '#C47FFF']))
        fig2.update_layout(**L(title='Own vs Neighbour Importance Split'))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-header'>Node-Level Explanations</div>", unsafe_allow_html=True)
    nt = st.selectbox("Select node type", ['Channel', 'Video', 'User', 'Comment'])
    exps = {
        'Channel': {'node': 'Aaj TV Official', 'conf': 0.998,
            'features': ['eigenvector', 'total_views', 'nbr_max_eigenvector', 'pagerank', 'nbr_mean_eigenvector'],
            'contribs': [-7.9004, 1.4318, -1.2264, 1.0057, -0.8332]},
        'Video': {'node': 'Investing In Crypto? Stop. Watch This.', 'conf': 0.992,
            'features': ['eigenvector', 'pagerank', 'nbr_max_eigenvector', 'nbr_mean_eigenvector', 'nbr_max_total_views'],
            'contribs': [-7.9004, 4.8553, -1.2264, -0.8332, 0.7378]},
        'User': {'node': '@usaidra', 'conf': 1.000,
            'features': ['eigenvector', 'nbr_max_eigenvector', 'nbr_mean_eigenvector', 'pagerank', 'total_views'],
            'contribs': [-7.9004, -1.2264, -0.8332, 0.2865, -0.0733]},
        'Comment': {'node': 'I have noticed how writers are trying...', 'conf': 0.992,
            'features': ['eigenvector', 'nbr_max_eigenvector', 'nbr_mean_eigenvector', 'total_views', 'nbr_max_total_views'],
            'contribs': [-7.2960, -1.2109, -0.8212, -0.0733, -0.0553]},
    }
    exp = exps[nt]
    col3, col4 = st.columns([1, 2])
    with col3:
        st.markdown(f"""
        <div class='insight-box'>
        <b>Node:</b> {exp['node']}<br>
        <b>Predicted class:</b> {nt.lower()}<br>
        <b>Confidence:</b> {exp['conf']:.3f}
        </div>""", unsafe_allow_html=True)
    with col4:
        colors = ['#3DD68C' if c > 0 else '#FF6B9D' for c in exp['contribs']]
        fig3 = go.Figure(go.Bar(
            x=exp['contribs'], y=exp['features'],
            orientation='h', marker_color=colors))
        fig3.add_vline(x=0, line_color='#6B7A99')
        fig3.update_layout(**L(title=f'Feature Contributions — {nt}',
            xaxis=dict(gridcolor='#1E2D50', linecolor='#1E2D50',
                       zerolinecolor='#1E2D50', tickfont=dict(size=11), title='Contribution'),
            yaxis=dict(gridcolor='#1E2D50', linecolor='#1E2D50',
                       zerolinecolor='#1E2D50', tickfont=dict(size=10))))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<div class='section-header'>Confusion Matrix</div>", unsafe_allow_html=True)
    cm = [[60, 0, 0, 0], [0, 74, 0, 0], [0, 0, 514, 0], [0, 0, 2, 618]]
    labels = ['channel', 'video', 'user', 'comment']
    fig4 = px.imshow(cm,
                     labels=dict(x='Predicted', y='True', color='Count'),
                     x=labels, y=labels,
                     color_continuous_scale='Blues', text_auto=True,
                     title='XGNN Node Classification — Confusion Matrix')
    fig4.update_layout(**L())
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""
    <div class='insight-box'>
    <b>XGNN Insight:</b> 99.8% accuracy across 4 node types. Only 2 comments misclassified as users.
    Own features (total_views, eigenvector) and neighbour features (nbr_max_eigenvector) contribute
    almost equally (50.2% / 49.8%), confirming graph structure is essential for node classification.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 7 — DECISION SUPPORT
# ══════════════════════════════════════════════════════════════
elif page == "🎯 Decision Support":
    st.markdown("# Decision Support Framework")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    for col, val, label, color in [
        (c1, "7", "Total Recommendations",  "#4D9FFF"),
        (c2, "5", "High Confidence",        "#3DD68C"),
        (c3, "2", "Medium Confidence",      "#FF8C42"),
    ]:
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value' style='color:{color}'>{val}</div>
            <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Real vs Synthetic Distribution</div>", unsafe_allow_html=True)
    fig_img('decision_support_real_vs_synthetic.png',
            'Real vs Synthetic Dataset — Distribution Comparison')

    st.markdown("<div class='section-header'>Strategic Recommendations</div>", unsafe_allow_html=True)
    recs = [
        ("High",   "Risk Alert",           "Analyse 267 viral outlier videos",
         "267 videos exceed 2,058,828 views (3σ above mean of 75,030). Identify content patterns."),
        ("High",   "Posting Strategy",     "Schedule uploads at 10:00, 20:00, 12:00",
         "Peak engagement hours across 50,247 combined videos."),
        ("High",   "Posting Strategy",     "Publish on Wed, Tue, Thu",
         "Top engagement days. Weekend lift only +0.6%."),
        ("High",   "Audience Engagement",  "Boost comment rate (0.045% vs 0.5% target)",
         "End every video with a question. Pin a discussion starter in comments."),
        ("High",   "Channel Strategy",     "Replicate HUM TV engagement model",
         "HUM TV highest total views (11.9M). Drama format + regular upload cadence works."),
        ("Medium", "Content Optimisation", "Use 300–1,000 character descriptions",
         "Medium-length descriptions outperform short and long across all channels."),
        ("Medium", "Content Optimisation", "Increase posting to 3+ videos/week",
         "Higher upload frequency correlates with sustained monthly view growth rate."),
    ]
    for conf, cat, rec, detail in recs:
        color = '#3DD68C' if conf == 'High' else '#FF8C42'
        icon  = '🟢' if conf == 'High' else '🟡'
        st.markdown(f"""
        <div class='insight-box' style='border-left-color:{color}'>
        <div style='display:flex;justify-content:space-between;margin-bottom:6px'>
            <span style='color:{color};font-weight:600'>{icon} {conf} Confidence</span>
            <span style='color:#4D9FFF;font-size:11px'>{cat}</span>
        </div>
        <b style='color:#E8EAF0'>{rec}</b><br>
        <span style='color:#6B7A99;font-size:12px'>{detail}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Channel Comparison</div>", unsafe_allow_html=True)
    # Compute from actual data
    if not videos.empty and 'channel_name' in videos.columns:
        ch = videos.groupby('channel_name').agg(
            avg_views=('views', 'mean'),
            avg_engagement=('engagement_rate', 'mean'),
            count=('video_id', 'count'),
        ).reset_index().rename(columns={
            'channel_name': 'Channel',
            'avg_views': 'Avg Views',
            'avg_engagement': 'Engagement',
            'count': 'Videos',
        })
    else:
        ch = pd.DataFrame({
            'Channel':    ['Aaj TV Official', 'Geo Entertainment', 'HUM TV', 'Raftar'],
            'Avg Views':  [1726,              65631,               118609,   44075],
            'Engagement': [0.0122,            0.0157,              0.0155,   0.0515],
        })

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(ch, x='Channel', y='Avg Views', color='Channel',
                     color_discrete_map=CH_COLORS, title='Avg Views per Video (Real Data)')
        fig.update_traces(showlegend=False)
        fig.update_layout(**L())
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.bar(ch, x='Channel', y='Engagement', color='Channel',
                      color_discrete_map=CH_COLORS, title='Avg Engagement Rate (Real Data)')
        fig2.update_traces(showlegend=False)
        top_eng = ch.loc[ch['Engagement'].idxmax()]
        fig2.add_annotation(
            x=top_eng['Channel'], y=top_eng['Engagement'],
            text=f"⭐ {top_eng['Engagement']:.4f}",
            showarrow=True, arrowhead=2,
            font=dict(color='#FF8C42', size=11),
            bgcolor='#1A2340', bordercolor='#FF8C42')
        fig2.update_layout(**L())
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <div class='insight-box'>
    <b>Key Finding:</b> Raftar achieves the highest engagement rate despite fewest real videos.
    HUM TV dominates total views. Comment rate (0.045%) vs benchmark (0.5%) is the biggest
    actionable gap — audience interaction, not reach, is the primary growth lever.
    </div>""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#2A3350;font-size:11px;padding:8px'>
Digital Media Analytics · Q13 Visualization &amp; Interpretability ·
Aaj TV · HUM TV · Geo Entertainment · Raftar
</div>""", unsafe_allow_html=True)
