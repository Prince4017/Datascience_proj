"""
Parcl Real Estate — Buyer Segmentation ML Pipeline
====================================================
Runs end-to-end: data loading → cleaning → encoding → scaling → KMeans → saves results.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────
print("[1/6] Loading data...")
import os
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
clients = pd.read_csv(os.path.join(DATA_DIR, "clients.csv"))
props   = pd.read_csv(os.path.join(DATA_DIR, "properties.csv"))
print(f"  ✓ Clients: {len(clients)} rows | Properties: {len(props)} rows")

# ─────────────────────────────────────────────
# STEP 2 — CLEAN DATA
# ─────────────────────────────────────────────
print("[2/6] Cleaning data...")

# Clean sale price — remove $, commas → float
props['sale_price_clean'] = (
    props['sale_price'].str.replace('$', '', regex=False)
                       .str.replace(',', '', regex=False)
                       .astype(float)
)

# Keep only Sold properties to join with clients
sold = props[props['listing_status'] == 'Sold'].copy()

# Merge clients with their purchased property
df = clients.merge(
    sold[['client_ref','sale_price_clean','floor_area_sqft','unit_category','tower_number']],
    left_on='client_id', right_on='client_ref', how='left'
)

# Calculate age from date_of_birth
def parse_dob(dob):
    for fmt in ['%m-%d-%Y', '%m/%d/%Y', '%d-%m-%Y']:
        try:
            return datetime.strptime(str(dob).strip(), fmt)
        except:
            pass
    return None

reference_date = datetime(2024, 6, 1)
df['dob_parsed'] = df['date_of_birth'].apply(parse_dob)
df['age'] = df['dob_parsed'].apply(
    lambda x: (reference_date - x).days // 365 if pd.notnull(x) else None
)

# Fill missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['sale_price_clean'].fillna(df['sale_price_clean'].median(), inplace=True)
df['floor_area_sqft'].fillna(df['floor_area_sqft'].median(), inplace=True)
df['unit_category'].fillna('Unknown', inplace=True)

# Remove duplicates
df.drop_duplicates(subset='client_id', keep='first', inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"  ✓ Final dataset: {len(df)} rows, {df.shape[1]} columns")
print(f"  ✓ Null values remaining: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────
# STEP 3 — FEATURE ENCODING
# ─────────────────────────────────────────────
print("[3/6] Encoding features...")

le = LabelEncoder()
categorical_cols = ['client_type', 'acquisition_purpose', 'loan_applied',
                    'referral_channel', 'unit_category']

for col in categorical_cols:
    df[col + '_enc'] = le.fit_transform(df[col])
    print(f"  ✓ Encoded: {col}")

# ─────────────────────────────────────────────
# STEP 4 — FEATURE SCALING
# ─────────────────────────────────────────────
print("[4/6] Scaling features...")

feature_cols = [
    'age', 'satisfaction_score', 'sale_price_clean', 'floor_area_sqft',
    'client_type_enc', 'acquisition_purpose_enc', 'loan_applied_enc', 'referral_channel_enc'
]

X = df[feature_cols].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"  ✓ Scaled {len(feature_cols)} features using StandardScaler")

# ─────────────────────────────────────────────
# STEP 5 — OPTIMAL CLUSTER SELECTION
# ─────────────────────────────────────────────
print("[5/6] Finding optimal K (Elbow + Silhouette)...")

inertias, sil_scores = [], []
k_range = range(2, 9)

for k in k_range:
    km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_temp.fit(X_scaled)
    inertias.append(km_temp.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km_temp.labels_))
    print(f"    K={k} | Inertia={km_temp.inertia_:.0f} | Silhouette={sil_scores[-1]:.4f}")

# Plot Elbow and Silhouette
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.patch.set_facecolor('#0f1117')
for ax in (ax1, ax2):
    ax.set_facecolor('#1e2130')
    for sp in ax.spines.values(): sp.set_color('#333')
    ax.tick_params(colors='white')

ax1.plot(list(k_range), inertias, 'o-', color='#4ECDC4', linewidth=2, markersize=8)
ax1.axvline(x=4, color='#FF6B6B', linestyle='--', label='Chosen K=4')
ax1.set_xlabel('K', color='white'); ax1.set_ylabel('Inertia', color='white')
ax1.set_title('Elbow Method', color='white', fontweight='bold')
ax1.legend(facecolor='#2a2d3e', labelcolor='white')

ax2.plot(list(k_range), sil_scores, 's-', color='#45B7D1', linewidth=2, markersize=8)
ax2.axvline(x=4, color='#FF6B6B', linestyle='--', label='Chosen K=4')
ax2.set_xlabel('K', color='white'); ax2.set_ylabel('Silhouette Score', color='white')
ax2.set_title('Silhouette Score', color='white', fontweight='bold')
ax2.legend(facecolor='#2a2d3e', labelcolor='white')

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'elbow_plot.png'), dpi=150, bbox_inches='tight', facecolor='#0f1117')
print("  ✓ Elbow plot saved → elbow_plot.png")

# ─────────────────────────────────────────────
# STEP 6 — KMEANS CLUSTERING (K=4)
# ─────────────────────────────────────────────
print("[6/6] Running KMeans with K=4...")

K = 4
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Map cluster numbers to business segment names
# (based on characteristic analysis)
CLUSTER_MAP = {
    0: 'C3 - Corporate Buyers',
    1: 'C2 - First-Time Buyers',
    2: 'C4 - Luxury Investors',
    3: 'C1 - Global Investors'
}
df['segment'] = df['cluster'].map(CLUSTER_MAP)

# Print summary
print("\n  ✓ Cluster Summary:")
summary = df.groupby('segment').agg(
    Count        = ('client_id',   'count'),
    Avg_Age      = ('age',          'mean'),
    Avg_Price    = ('sale_price_clean', 'mean'),
    Pct_Loan     = ('loan_applied', lambda x: (x=='Yes').mean()*100),
    Pct_Invest   = ('acquisition_purpose', lambda x: (x=='Investment').mean()*100),
).round(1)
print(summary.to_string())

# Save enriched dataset
df.to_csv(os.path.join(DATA_DIR, 'clustered_clients.csv'), index=False)
print("\n  ✓ Results saved → clustered_clients.csv")
print("\n✅ Pipeline complete!")
