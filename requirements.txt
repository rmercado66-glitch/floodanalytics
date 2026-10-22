import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, silhouette_score
)

st.set_page_config(
    page_title="Flood Analytics Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    }
    .main-header {
        font-size: 2.0rem; font-weight: 700; color: #004D40;
        text-align: center; padding: 0.6rem 0 0.2rem 0;
        letter-spacing: 0.4px;
    }
    .sub-header {
        font-size: 0.95rem; color: #546E7A;
        text-align: center; margin-bottom: 1.6rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #004D40 0%, #00796B 100%);
        padding: 1.15rem; border-radius: 10px; color: #FFFFFF;
        text-align: center; box-shadow: 0 3px 12px rgba(0,77,64,0.22);
        border-bottom: 3px solid #FFC107;
    }
    .metric-card h2 { color: #FFE082; margin: 0; font-size: 1.75rem; font-weight: 700; }
    .metric-card p  { color: #E0F2F1; margin: 0; font-size: 0.82rem;
                      text-transform: uppercase; letter-spacing: 0.8px; }

    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #ECEFF1; border-radius: 6px 6px 0 0;
        padding: 10px 16px; font-weight: 600; color: #004D40;
    }
    .stTabs [aria-selected="true"] {
        background-color: #004D40 !important; color: #FFE082 !important;
    }
    .dataframe thead tr th { background-color: #E0F2F1 !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=True)
def load_flood_data():
    search_paths = ['.', '/content/data', '/content', '/mount/src']
    possible_filenames = ['FloodData.xlsx', 'FloodData.xls', 'FloodData.csv', 'flood_data.xlsx', 'flood_data.csv', 'Flood Data.xlsx']
    filepath = None
    for path in search_paths:
        if not os.path.isdir(path): continue
        for fname in possible_filenames:
            full_path = os.path.join(path, fname)
            if os.path.exists(full_path):
                filepath = full_path
                break
        if filepath: break
    if filepath is None: return None, "No dataset found."
    try:
        df = pd.read_csv(filepath) if filepath.lower().endswith('.csv') else pd.read_excel(filepath, header=None)
        return df, f"Loaded: {os.path.basename(filepath)}"
    except Exception as e:
        return None, f"Error loading file: {e}"

def clean_text(value):
    if pd.isna(value): return np.nan
    s = str(value).replace('\xa0', ' ').replace('\u200b', '').replace('\t', ' ').replace('\n', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    if s.lower() in {'', 'nan', 'none', 'n/a', '-', '--', '?', 'undefined'}: return np.nan
    return s

MONTH_MAP = {'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6, 'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
             'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12}

def parse_month(value):
    s = clean_text(value)
    if pd.isna(s): return np.nan
    s = str(s).lower()
    if s.isdigit():
        m = int(s)
        return m if 1 <= m <= 12 else np.nan
    return MONTH_MAP.get(s, np.nan)

def parse_number(value):
    s = clean_text(value)
    if pd.isna(s): return np.nan
    m = re.search(r'(\d+(?:\.\d+)?)', str(s))
    return float(m.group(1)) if m else np.nan

def parse_damage(value):
    s = clean_text(value)
    if pd.isna(s): return np.nan
    s = str(s).replace(' ', '').replace(',', '').replace('php', '').replace('PHP', '')
    if s.count('.') > 1:
        parts = s.split('.')
        s = ''.join(parts[:-1]) + '.' + parts[-1]
    s = re.sub(r'[^\d.]', '', s)
    try: return float(s)
    except: return np.nan

@st.cache_data(show_spinner=True)
def preprocess_data(raw):
    df = raw.copy()
    df.columns = [clean_text(c) for c in df.iloc[0]]
    df = df.iloc[1:].reset_index(drop=True)
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index] = [f"{dup}_{i}" if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols

    clean = pd.DataFrame()
    clean['Year'] = pd.to_numeric(df.get('Year', np.nan), errors='coerce').ffill()
    clean['Month_Num'] = df.get('Month', pd.Series()).apply(parse_month).ffill()
    clean['Barangay'] = df.get('Barangay', pd.Series()).apply(lambda x: clean_text(x).title() if pd.notna(x) else np.nan)
    clean['Flood_Cause'] = df.get('Flood Cause', pd.Series()).apply(lambda x: clean_text(x) if pd.notna(x) else 'Unknown')
    clean['Water_Level_Ft'] = df.get('Water Level', pd.Series()).apply(parse_number)
    clean['Families_Affected'] = pd.to_numeric(df.get('No. of Families affected', pd.Series()).apply(clean_text), errors='coerce')
    clean['Damage_Infrastructure'] = df.get('Damage Infrastructure', pd.Series()).apply(parse_damage)
    clean['Damage_Agriculture'] = df.get('Damage Agriculture', pd.Series()).apply(parse_damage)

    clean = clean.dropna(subset=['Barangay'])
    clean = clean[clean['Barangay'] != 'Unknown']

    MONTH_NAMES = {1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}
    clean['Month_Name'] = clean['Month_Num'].map(MONTH_NAMES)
    clean['Season'] = clean['Month_Num'].apply(lambda m: 'Dry Season' if m in (12,1,2,3,4,5) else ('Wet Season' if pd.notna(m) else 'Unknown'))
    clean['Total_Damage'] = clean['Damage_Infrastructure'].fillna(0) + clean['Damage_Agriculture'].fillna(0)

    def severity_level(row):
        wl, fam, dmg = row['Water_Level_Ft'], row['Families_Affected'], row['Total_Damage']
        high_count = 0
        low_count = 0
        if pd.notna(wl):
            if wl > 4.0: high_count += 1
            if wl <= 2.0: low_count += 1
        if pd.notna(fam):
            if fam > 200: high_count += 1
            if fam < 50: low_count += 1
        if pd.notna(dmg):
            if dmg > 500_000: high_count += 1
            if dmg < 100_000: low_count += 1
        if high_count >= 2: return 'High'
        if low_count == 3: return 'Low'
        return 'Moderate'
    
    clean['Severity'] = clean.apply(severity_level, axis=1)
    return clean

def simple_apriori(transactions, min_support=0.1):
    n = len(transactions)
    if n == 0: return {}
    items = sorted(set(item for t in transactions for item in t))
    freq = {}
    for item in items:
        count = sum(1 for t in transactions if item in t)
        support = count / n
        if support >= min_support: freq[frozenset([item])] = support
    all_freq = dict(freq)
    k = 2
    while freq:
        freq_list = list(freq.keys())
        candidates = set()
        for i in range(len(freq_list)):
            for j in range(i + 1, len(freq_list)):
                union = freq_list[i] | freq_list[j]
                if len(union) == k: candidates.add(union)
        new_freq = {}
        for cand in candidates:
            count = sum(1 for t in transactions if cand.issubset(t))
            support = count / n
            if support >= min_support: new_freq[cand] = support
        all_freq.update(new_freq)
        freq = new_freq
        k += 1
    return all_freq

def generate_rules(freq_itemsets, min_confidence=0.5):
    rules = []
    for itemset, support in freq_itemsets.items():
        if len(itemset) < 2: continue
        for item in itemset:
            antecedent = frozenset([item])
            consequent = itemset - antecedent
            if not consequent: continue
            ant_support = freq_itemsets.get(antecedent, 0)
            if ant_support <= 0: continue
            confidence = support / ant_support
            if confidence < min_confidence: continue
            cons_support = freq_itemsets.get(consequent, 0)
            lift = confidence / cons_support if cons_support > 0 else 0
            rules.append({'antecedent': set(antecedent), 'consequent': set(consequent), 'support': support, 'confidence': confidence, 'lift': lift})
    seen = set()
    unique_rules = []
    for r in rules:
        key = (frozenset(r['antecedent']), frozenset(r['consequent']))
        if key in seen: continue
        seen.add(key)
        unique_rules.append(r)
    return unique_rules

raw, load_message = load_flood_data()

st.markdown('<div class="main-header">Flood Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Historical Flood Records Analysis</div>', unsafe_allow_html=True)

if raw is None:
    st.error(f"Error: {load_message}")
    st.stop()

df = preprocess_data(raw)

st.sidebar.header("Dashboard Controls")
brgy_options = sorted(df['Barangay'].dropna().unique())
barangay_filter = st.sidebar.multiselect("Filter by Barangay:", options=brgy_options, default=brgy_options)

year_min = int(df['Year'].dropna().min()) if df['Year'].notna().any() else 2015
year_max = int(df['Year'].dropna().max()) if df['Year'].notna().any() else 2025
year_range = st.sidebar.slider("Year Range", year_min, year_max, (year_min, year_max))

st.sidebar.markdown("---")
st.sidebar.subheader("Model Parameters")
k_clusters = st.sidebar.slider("K-Means Clusters", 2, 6, 3)
min_support = st.sidebar.slider("Apriori Min Support", 0.01, 0.5, 0.05, 0.01)
min_confidence = st.sidebar.slider("Apriori Min Confidence", 0.1, 1.0, 0.3, 0.05)

df_f = df[(df['Barangay'].isin(barangay_filter)) & (df['Year'].between(year_range[0], year_range[1]))].copy()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview", "High-Risk Barangays", "Seasonal Patterns", "Severity Levels", "Association Rules", "Decision Support"
])

with tab1:
    st.header("Overview of Historical Flood Records")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><p>Total Flood Records</p><h2>{len(df_f):,}</h2></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><p>Distinct Barangays</p><h2>{df_f["Barangay"].nunique()}</h2></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><p>Families Affected</p><h2>{int(df_f["Families_Affected"].sum()):,}</h2></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><p>Total Damage (PHP)</p><h2>{int(df_f["Total_Damage"].sum()):,}</h2></div>', unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Flood Records by Barangay")
        fig, ax = plt.subplots(figsize=(10, 5))
        df_f['Barangay'].value_counts().plot(kind='bar', color='#00796B', edgecolor='black', ax=ax)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
    with col_b:
        st.subheader("Flood Cause Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        df_f['Flood_Cause'].value_counts().head(10).plot(kind='pie', autopct='%1.1f%%', cmap='Set3', ax=ax)
        st.pyplot(fig)

with tab2:
    st.header("High-Risk Flood Barangays")
    agg = df_f.groupby('Barangay').agg(
        Total_Records=('Year', 'count'),
        Mean_Water_Level=('Water_Level_Ft', 'mean'),
        Max_Water_Level=('Water_Level_Ft', 'max'),
        Total_Families=('Families_Affected', 'sum'),
        Mean_Families=('Families_Affected', 'mean'),
        Total_Damage=('Total_Damage', 'sum'),
        Severe_Count=('Severity', lambda x: (x == 'High').sum())
    ).fillna(0)

    if len(agg) >= k_clusters * 2:
        feature_cols = ['Mean_Water_Level', 'Max_Water_Level', 'Mean_Families', 'Total_Families', 'Total_Damage', 'Total_Records', 'Severe_Count']
        X = agg[feature_cols].copy()
        X = X.loc[:, X.std() > 0]
        X_scaled = StandardScaler().fit_transform(X)
        
        km = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0.0
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Clusters", k_clusters)
        c2.metric("Silhouette Score", f"{sil:.4f}")
        c3.metric("Within-Cluster SS", f"{km.inertia_:,.2f}")
        
        agg['Cluster'] = labels
        cluster_rank = agg.groupby('Cluster')['Total_Damage'].mean().sort_values(ascending=False)
        rank_map = {c: f"Tier {i+1}" for i, c in enumerate(cluster_rank.index)}
        agg['Risk_Tier'] = agg['Cluster'].map(rank_map)

        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            counts = pd.Series(labels).value_counts().sort_index()
            colors = plt.cm.Set2(np.linspace(0, 1, k_clusters))
            ax.bar([f"Cluster {i}" for i in counts.index], counts.values, color=colors[:len(counts)], edgecolor='black')
            ax.set_ylabel("Barangays")
            ax.set_title("Barangays per Cluster")
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=80, alpha=0.85, edgecolors='black')
            plt.colorbar(sc, label='Cluster')
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("Barangays in PCA Space")
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("---")
        st.subheader("Barangay Risk Ranking")
        ranked = agg.sort_values(['Risk_Tier', 'Total_Damage'], ascending=[True, False])
        st.dataframe(ranked[['Risk_Tier', 'Cluster', 'Total_Records', 'Mean_Water_Level', 'Max_Water_Level', 'Mean_Families', 'Total_Families', 'Total_Damage', 'Severe_Count']].round(2), use_container_width=True)
    else:
        st.warning("Not enough barangays for stable clustering.")

with tab3:
    st.header("Seasonal Flood Patterns")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Flood Occurrences by Month")
        m_order = ['January','February','March','April','May','June','July','August','September','October','November','December']
        m_counts = df_f['Month_Name'].value_counts().reindex(m_order).fillna(0)
        fig, ax = plt.subplots(figsize=(8, 5))
        m_counts.plot(kind='bar', color='#26A69A', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    with col2:
        st.subheader("Flood Occurrences by Season")
        s_counts = df_f['Season'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#26A69A' if s == 'Wet Season' else '#FFB74D' for s in s_counts.index]
        s_counts.plot(kind='bar', color=colors, edgecolor='black', ax=ax)
        plt.xticks(rotation=0)
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Heatmap: Flood Occurrence by Year and Month")
    pivot = df_f.pivot_table(index='Month_Name', columns='Year', values='Water_Level_Ft', aggfunc='count', fill_value=0).reindex(m_order).fillna(0)
    if pivot.size > 0:
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', linewidths=0.5, ax=ax)
        plt.tight_layout()
        st.pyplot(fig)

with tab4:
    st.header("Severity Levels")
    mdf = df_f.dropna(subset=['Water_Level_Ft', 'Families_Affected', 'Total_Damage', 'Severity'])
    if len(mdf) > 15 and mdf['Severity'].nunique() >= 2:
        Xm = mdf[['Water_Level_Ft', 'Families_Affected', 'Total_Damage']].values
        ym = mdf['Severity'].values
        le = LabelEncoder()
        y_enc = le.fit_transform(ym)
        
        min_class_count = pd.Series(y_enc).value_counts().min()
        stratify_arg = y_enc if min_class_count >= 2 else None
        
        X_train, X_test, y_train, y_test = train_test_split(Xm, y_enc, test_size=0.30, random_state=42, stratify=stratify_arg)
        dt = DecisionTreeClassifier(max_depth=4, random_state=42, min_samples_leaf=2).fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        c2.metric("Precision", f"{precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
        c3.metric("Recall", f"{recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
        c4.metric("F1-Score", f"{f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
        
        st.markdown("---")
        st.subheader("Decision Tree Visualization")
        fig, ax = plt.subplots(figsize=(16, 8))
        plot_tree(dt, feature_names=['Water_Level_Ft', 'Families_Affected', 'Total_Damage'], class_names=list(le.classes_), filled=True, rounded=True, ax=ax, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Not enough data for Decision Tree.")

with tab5:
    st.header("Association Rules (Apriori)")
    txn_df = df_f.dropna(subset=['Barangay'])
    transactions = []
    for _, row in txn_df.iterrows():
        txn = [f"Barangay={row['Barangay']}", f"Cause={row['Flood_Cause']}", f"Severity={row['Severity']}"]
        if pd.notna(row['Season']) and row['Season'] != 'Unknown': txn.append(f"Season={row['Season']}")
        if pd.notna(row['Month_Name']): txn.append(f"Month={row['Month_Name']}")
        transactions.append(set(txn))

    if len(transactions) >= 5:
        freq = simple_apriori(transactions, min_support=min_support)
        rules = generate_rules(freq, min_confidence=min_confidence)

        c1, c2, c3 = st.columns(3)
        c1.metric("Frequent Itemsets", len(freq))
        c2.metric("Rules Mined", len(rules))
        c3.metric("Transactions", len(transactions))

        if rules:
            rules_df = pd.DataFrame([{'Antecedent': ', '.join(sorted(r['antecedent'])), 'Consequent': ', '.join(sorted(r['consequent'])), 'Support': round(r['support'], 4), 'Confidence': round(r['confidence'], 4), 'Lift': round(r['lift'], 4)} for r in rules]).sort_values(['Lift', 'Confidence'], ascending=False)
            st.dataframe(rules_df.head(30), use_container_width=True)

            st.markdown("---")
            st.subheader("Support vs Confidence Scatter")
            fig, ax = plt.subplots(figsize=(9, 5))
            sc = ax.scatter(rules_df['Support'], rules_df['Confidence'], c=rules_df['Lift'], cmap='YlOrRd', s=80, alpha=0.85, edgecolors='black')
            plt.colorbar(sc, label='Lift')
            ax.set_xlabel("Support")
            ax.set_ylabel("Confidence")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No rules met the current thresholds. Try lowering the sliders in the sidebar.")

with tab6:
    st.header("Decision Support")
    top_brgy = df_f['Barangay'].value_counts().idxmax() if not df_f.empty else 'N/A'
    top_cause = df_f['Flood_Cause'].value_counts().idxmax() if not df_f.empty else 'N/A'
    top_month = df_f['Month_Name'].value_counts().idxmax() if df_f['Month_Name'].notna().any() else 'N/A'
    high_sev_pct = (df_f['Severity'] == 'High').sum() / max(len(df_f), 1) * 100

    st.markdown(f"**Concentration of Flood Risk:** A small group of barangays accounts for a disproportionate share of all recorded flood events. The top barangay is **{top_brgy}**. K-Means clustering further tiers these barangays into similar-risk groups (Tier 1 = highest damage profile).")
    st.markdown(f"**Seasonal and Monthly Concentration:** Flood occurrences are not uniformly distributed. The most frequently recorded flood month is **{top_month}**.")
    st.markdown(f"**Dominant Flood-Causing Weather System:** The most frequently recorded flood-causing weather system is **{top_cause}**.")
    st.markdown(f"**Severity Profile of Flood Events:** Of all filtered flood records, **{high_sev_pct:.1f}%** are classified as High severity. Decision tree classification identified water level, family count, and total damage as the primary discriminators between severity tiers.")
