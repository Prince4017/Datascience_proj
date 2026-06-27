"""
Parcl Real Estate — Buyer Segmentation Dashboard
=================================================
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Parcl — Buyer Intelligence",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2a2d3e);
        border-radius: 12px; padding: 20px; border-left: 4px solid;
        margin: 8px 0;
    }
    h1 { color: #ffffff !important; }
    .stMetric label { color: #aaa !important; font-size: 0.85rem; }
    .stMetric > div > div { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD & PROCESS DATA (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_and_cluster():
    clients = pd.read_csv("clients.csv")
    props   = pd.read_csv("properties.csv")

    props['sale_price_clean'] = (
        props['sale_price'].str.replace('$','', regex=False)
                           .str.replace(',','', regex=False)
                           .astype(float)
    )
    sold = props[props['listing_status'] == 'Sold'].copy()
    df = clients.merge(
        sold[['client_ref','sale_price_clean','floor_area_sqft','unit_category']],
        left_on='client_id', right_on='client_ref', how='left'
    )

    def parse_dob(dob):
        for fmt in ['%m-%d-%Y','%m/%d/%Y','%d-%m-%Y']:
            try: return datetime.strptime(str(dob).strip(), fmt)
            except: pass
        return None

    ref = datetime(2024, 6, 1)
    df['dob_parsed'] = df['date_of_birth'].apply(parse_dob)
    df['age'] = df['dob_parsed'].apply(lambda x: (ref - x).days // 365 if pd.notnull(x) else None)
    df['age'].fillna(df['age'].median(), inplace=True)
    df['sale_price_clean'].fillna(df['sale_price_clean'].median(), inplace=True)
    df['floor_area_sqft'].fillna(df['floor_area_sqft'].median(), inplace=True)
    df['unit_category'].fillna('Unknown', inplace=True)
    df.drop_duplicates(subset='client_id', keep='first', inplace=True)

    le = LabelEncoder()
    for col in ['client_type','acquisition_purpose','loan_applied','referral_channel','unit_category']:
        df[col+'_enc'] = le.fit_transform(df[col])

    feature_cols = ['age','satisfaction_score','sale_price_clean','floor_area_sqft',
                    'client_type_enc','acquisition_purpose_enc','loan_applied_enc','referral_channel_enc']
    X_scaled = StandardScaler().fit_transform(df[feature_cols])

    df['cluster'] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X_scaled)

    CLUSTER_MAP = {0:'C3 - Corporate Buyers', 1:'C2 - First-Time Buyers',
                   2:'C4 - Luxury Investors', 3:'C1 - Global Investors'}
    CLUSTER_COLORS = {0:'#4e79a7', 1:'#f28e2b', 2:'#e15759', 3:'#76b7b2'}
    df['segment'] = df['cluster'].map(CLUSTER_MAP)
    df['color'] = df['cluster'].map(CLUSTER_COLORS)
    return df

df = load_and_cluster()

SEGMENT_COLORS = {
    'C3 - Corporate Buyers':  '#4e79a7',
    'C2 - First-Time Buyers': '#f28e2b',
    'C4 - Luxury Investors':  '#e15759',
    'C1 - Global Investors':  '#76b7b2',
}

# ─────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────
st.sidebar.image("https://via.placeholder.com/200x50/1e2130/ffffff?text=Parcl+Co.", use_column_width=True)
st.sidebar.markdown("---")
st.sidebar.header("🔍 Filter Data")

countries = ['All'] + sorted(df['country'].unique().tolist())
sel_country = st.sidebar.selectbox("🌍 Country", countries)

regions = ['All'] + sorted(df['region'].unique().tolist())
sel_region = st.sidebar.selectbox("📍 Region", regions)

purposes = ['All'] + sorted(df['acquisition_purpose'].unique().tolist())
sel_purpose = st.sidebar.selectbox("🎯 Acquisition Purpose", purposes)

types = ['All'] + sorted(df['client_type'].unique().tolist())
sel_type = st.sidebar.selectbox("👤 Client Type", types)

segments = ['All'] + sorted(df['segment'].unique().tolist())
sel_segment = st.sidebar.selectbox("🏷️ Segment", segments)

st.sidebar.markdown("---")
st.sidebar.markdown("**📊 Model Info**")
st.sidebar.markdown("Algorithm: `KMeans (K=4)`")
st.sidebar.markdown("Features: `8 variables`")
st.sidebar.markdown("Dataset: `7,305 transactions`")

# ─────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────
filtered = df.copy()
if sel_country  != 'All': filtered = filtered[filtered['country'] == sel_country]
if sel_region   != 'All': filtered = filtered[filtered['region'] == sel_region]
if sel_purpose  != 'All': filtered = filtered[filtered['acquisition_purpose'] == sel_purpose]
if sel_type     != 'All': filtered = filtered[filtered['client_type'] == sel_type]
if sel_segment  != 'All': filtered = filtered[filtered['segment'] == sel_segment]

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 🏢 Parcl — Buyer Segmentation Intelligence")
st.markdown("*AI-powered buyer profiling for smarter real estate marketing*")
st.markdown("---")

# ─────────────────────────────────────────────
# KPI METRICS ROW
# ─────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Buyers", f"{len(filtered):,}")
k2.metric("Avg Sale Price", f"${filtered['sale_price_clean'].mean():,.0f}")
k3.metric("Avg Age", f"{filtered['age'].mean():.0f} yrs")
k4.metric("Loan Rate", f"{(filtered['loan_applied']=='Yes').mean()*100:.1f}%")
k5.metric("Inv. Purpose", f"{(filtered['acquisition_purpose']=='Investment').mean()*100:.1f}%")

st.markdown("---")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Segment Overview",
    "💰 Investment Behavior",
    "🗺️ Geographic Analysis",
    "🔬 Segment Insights"
])

# ══════════════════════════════════════════════
# TAB 1 — SEGMENT OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    st.subheader("Buyer Segment Distribution")
    col1, col2 = st.columns([1, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#0f1117')
        seg_counts = filtered['segment'].value_counts()
        colors = [SEGMENT_COLORS.get(s, '#888') for s in seg_counts.index]
        wedges, texts, autos = ax.pie(
            seg_counts.values, labels=seg_counts.index, colors=colors,
            autopct='%1.1f%%', startangle=90,
            textprops={'color':'white','fontsize':9}
        )
        for a in autos: a.set_fontsize(8)
        ax.set_title('Segment Distribution', color='white', fontsize=13, fontweight='bold', pad=15)
        st.pyplot(fig)

    with col2:
        # Summary table
        st.markdown("#### Cluster Statistics")
        stats = filtered.groupby('segment').agg(
            Count        = ('client_id', 'count'),
            Avg_Age      = ('age', 'mean'),
            Avg_Price    = ('sale_price_clean', 'mean'),
            Avg_Sat      = ('satisfaction_score', 'mean'),
            Loan_Pct     = ('loan_applied', lambda x: f"{(x=='Yes').mean()*100:.1f}%"),
            Invest_Pct   = ('acquisition_purpose', lambda x: f"{(x=='Investment').mean()*100:.1f}%"),
        ).round(1)
        stats.columns = ['Count','Avg Age','Avg Price ($)','Avg Sat.','Loan %','Invest %']
        st.dataframe(stats, use_container_width=True)

        # Segment size bar
        st.markdown("#### Buyers per Segment")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        fig2.patch.set_facecolor('#0f1117'); ax2.set_facecolor('#1e2130')
        bars = ax2.barh(seg_counts.index, seg_counts.values,
                        color=[SEGMENT_COLORS.get(s,'#888') for s in seg_counts.index])
        for bar, val in zip(bars, seg_counts.values):
            ax2.text(bar.get_width()+20, bar.get_y()+bar.get_height()/2,
                     str(val), va='center', color='white', fontsize=9)
        ax2.tick_params(colors='white', labelsize=8)
        for sp in ax2.spines.values(): sp.set_color('#333')
        ax2.set_xlabel('Number of Buyers', color='white')
        ax2.set_title('', color='white')
        st.pyplot(fig2)

# ══════════════════════════════════════════════
# TAB 2 — INVESTMENT BEHAVIOR
# ══════════════════════════════════════════════
with tab2:
    st.subheader("Investment Behavior by Segment")
    col1, col2 = st.columns(2)

    with col1:
        # Avg price by segment
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#1e2130')
        price_data = filtered.groupby('segment')['sale_price_clean'].mean().sort_values()
        colors = [SEGMENT_COLORS.get(s,'#888') for s in price_data.index]
        bars = ax.barh(price_data.index, price_data.values/1000, color=colors)
        for bar, val in zip(bars, price_data.values):
            ax.text(bar.get_width()+2, bar.get_y()+bar.get_height()/2,
                    f'${val/1000:.0f}K', va='center', color='white', fontsize=9, fontweight='bold')
        ax.tick_params(colors='white', labelsize=8)
        for sp in ax.spines.values(): sp.set_color('#333')
        ax.set_xlabel('Avg Sale Price ($K)', color='white')
        ax.set_title('Average Sale Price by Segment', color='white', fontweight='bold')
        st.pyplot(fig)

    with col2:
        # Loan vs No Loan stacked
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#1e2130')
        loan_yes = filtered.groupby('segment').apply(lambda x: (x['loan_applied']=='Yes').mean()*100)
        loan_no  = 100 - loan_yes
        y = np.arange(len(loan_yes))
        ax.barh(y, loan_yes.values, color='#4CAF50', label='Loan Applied', alpha=0.85)
        ax.barh(y, loan_no.values,  left=loan_yes.values, color='#F44336', label='No Loan', alpha=0.85)
        ax.set_yticks(y); ax.set_yticklabels(loan_yes.index, color='white', fontsize=8)
        ax.tick_params(colors='white')
        for sp in ax.spines.values(): sp.set_color('#333')
        ax.legend(facecolor='#2a2d3e', labelcolor='white', fontsize=8)
        ax.set_xlabel('Percentage (%)', color='white')
        ax.set_title('Loan Behavior by Segment', color='white', fontweight='bold')
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        # Referral channel
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#1e2130')
        ch_data = pd.crosstab(filtered['segment'], filtered['referral_channel'], normalize='index')*100
        x = np.arange(len(ch_data)); w = 0.28
        for i, (col_name, clr) in enumerate(zip(ch_data.columns, ['#FF6B6B','#4ECDC4','#45B7D1'])):
            ax.bar(x + i*w, ch_data[col_name].values, w, label=col_name, color=clr, alpha=0.85)
        ax.set_xticks(x + w); ax.set_xticklabels(ch_data.index, color='white', fontsize=7, rotation=15)
        ax.tick_params(colors='white')
        for sp in ax.spines.values(): sp.set_color('#333')
        ax.legend(facecolor='#2a2d3e', labelcolor='white', fontsize=8)
        ax.set_ylabel('Percentage (%)', color='white')
        ax.set_title('Referral Channel by Segment', color='white', fontweight='bold')
        st.pyplot(fig)

    with col4:
        # Age vs Price scatter
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#1e2130')
        for seg, color in SEGMENT_COLORS.items():
            sub = filtered[filtered['segment']==seg].sample(min(200, len(filtered[filtered['segment']==seg])), random_state=42)
            ax.scatter(sub['age'], sub['sale_price_clean']/1000,
                       c=color, alpha=0.5, s=15, label=seg.split(' - ')[1])
        ax.tick_params(colors='white')
        for sp in ax.spines.values(): sp.set_color('#333')
        ax.legend(fontsize=7, facecolor='#2a2d3e', labelcolor='white')
        ax.set_xlabel('Age', color='white'); ax.set_ylabel('Price ($K)', color='white')
        ax.set_title('Age vs Sale Price', color='white', fontweight='bold')
        st.pyplot(fig)

# ══════════════════════════════════════════════
# TAB 3 — GEOGRAPHIC ANALYSIS
# ══════════════════════════════════════════════
with tab3:
    st.subheader("Geographic Buyer Distribution")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Top Countries by Buyer Count")
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#1e2130')
        top_c = filtered['country'].value_counts().head(10)
        ax.barh(top_c.index[::-1], top_c.values[::-1], color='#7986CB', edgecolor='none')
        ax.tick_params(colors='white', labelsize=9)
        for sp in ax.spines.values(): sp.set_color('#333')
        ax.set_xlabel('Buyer Count', color='white')
        ax.set_title('Top 10 Countries', color='white', fontweight='bold')
        st.pyplot(fig)

    with col2:
        st.markdown("#### Top Regions by Buyer Count")
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#1e2130')
        top_r = filtered['region'].value_counts().head(10)
        ax.barh(top_r.index[::-1], top_r.values[::-1], color='#4DB6AC', edgecolor='none')
        ax.tick_params(colors='white', labelsize=9)
        for sp in ax.spines.values(): sp.set_color('#333')
        ax.set_xlabel('Buyer Count', color='white')
        ax.set_title('Top 10 Regions', color='white', fontweight='bold')
        st.pyplot(fig)

    st.markdown("#### Segment Distribution by Country (Top 8)")
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#1e2130')
    top8_countries = filtered['country'].value_counts().head(8).index
    geo_data = pd.crosstab(
        filtered[filtered['country'].isin(top8_countries)]['country'],
        filtered[filtered['country'].isin(top8_countries)]['segment'],
        normalize='index'
    ) * 100
    x = np.arange(len(geo_data)); w = 0.2
    for i, (seg, color) in enumerate(SEGMENT_COLORS.items()):
        if seg in geo_data.columns:
            ax.bar(x + i*w, geo_data[seg].values, w,
                   label=seg.split(' - ')[1], color=color, alpha=0.85)
    ax.set_xticks(x + w*1.5); ax.set_xticklabels(geo_data.index, color='white', fontsize=9, rotation=20)
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_color('#333')
    ax.legend(facecolor='#2a2d3e', labelcolor='white', fontsize=8)
    ax.set_ylabel('Percentage (%)', color='white')
    ax.set_title('Buyer Segment Mix by Country', color='white', fontweight='bold')
    st.pyplot(fig)

# ══════════════════════════════════════════════
# TAB 4 — SEGMENT INSIGHTS
# ══════════════════════════════════════════════
with tab4:
    st.subheader("Detailed Segment Insights")

    seg_info = {
        'C1 - Global Investors': {
            'icon': '🌐', 'color': '#76b7b2',
            'desc': '100% investment-driven buyers. Spread across multiple countries with a focus on portfolio growth. Mid-range price point with strong international diversity.',
            'strategy': 'Target via international property expos, multi-currency payment options, ROI reports'
        },
        'C2 - First-Time Buyers': {
            'icon': '🏠', 'color': '#f28e2b',
            'desc': 'Largest segment — individuals purchasing homes for personal use. Lower price sensitivity, moderate loan usage. Website is their primary discovery channel.',
            'strategy': 'First-home education content, mortgage calculator tools, neighbourhood guides'
        },
        'C3 - Corporate Buyers': {
            'icon': '🏢', 'color': '#4e79a7',
            'desc': '100% corporate entities. Buying for both investment and business use. Larger units, mid-range pricing. Strongest candidates for bulk-purchase deals.',
            'strategy': 'B2B outreach, corporate account managers, portfolio pricing packages'
        },
        'C4 - Luxury Investors': {
            'icon': '💎', 'color': '#e15759',
            'desc': 'Highest average sale price ($490K+) and largest floor areas (1,608 sqft). Premium individual buyers seeking large units. Mix of investment and home use.',
            'strategy': 'VIP showings, premium listings, concierge service, off-plan launches'
        }
    }

    for seg, info in seg_info.items():
        seg_data = filtered[filtered['segment'] == seg]
        if len(seg_data) == 0:
            continue

        with st.expander(f"{info['icon']} {seg} — {len(seg_data)} buyers", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg Price",    f"${seg_data['sale_price_clean'].mean():,.0f}")
            c2.metric("Avg Age",      f"{seg_data['age'].mean():.0f} yrs")
            c3.metric("Loan Rate",    f"{(seg_data['loan_applied']=='Yes').mean()*100:.1f}%")
            c4.metric("Invest %",     f"{(seg_data['acquisition_purpose']=='Investment').mean()*100:.1f}%")

            st.markdown(f"**Profile:** {info['desc']}")
            st.markdown(f"**🎯 Marketing Strategy:** {info['strategy']}")

            # Top regions for this segment
            top_regions = seg_data['region'].value_counts().head(5)
            st.markdown(f"**Top Regions:** {', '.join([f'{r} ({n})' for r, n in zip(top_regions.index, top_regions.values)])}")

    st.markdown("---")
    st.subheader("Raw Data Explorer")
    show_cols = ['client_id','segment','client_type','country','region',
                 'acquisition_purpose','loan_applied','age','satisfaction_score',
                 'sale_price_clean','referral_channel']
    st.dataframe(filtered[show_cols].rename(columns={'sale_price_clean':'sale_price'}).head(500),
                 use_container_width=True)
    csv = filtered[show_cols].to_csv(index=False)
    st.download_button("⬇️ Download Filtered Data (CSV)", csv, "parcl_segmented_buyers.csv", "text/csv")

st.markdown("---")
st.markdown("*Parcl Co. Limited — Buyer Segmentation Intelligence | Built with Python, scikit-learn, Streamlit*")
