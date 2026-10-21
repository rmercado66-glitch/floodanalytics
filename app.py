import re
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score

st.set_page_config(layout="wide", page_title="Flood Analysis Dashboard", page_icon="🌊")

CONFIG = {
    "file_path": "FloodData.xlsx", 
    "barangay_col": "Barangay",
    "cause_col": "Flood Cause",
    "water_col": "Water Level",
    "families_col": "No. of Families affected",
    "infra_col": "Damage Infrastructure",
    "agri_col": "Damage Agriculture",
    "month_col": "Month",
    "year_col": "Year",
    "sheet_index": 0,
}

MONTH_MAP = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12}

def clean_text(v):
    if pd.isna(v): return v
    s = unicodedata.normalize("NFKC", str(v))
    s = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]", "", s)
    return re.sub(r"\s+", " ", s).strip()

def parse_number(v):
    if pd.isna(v): return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)): return float(v)
    s = clean_text(v).replace("ft.", "").replace("ft", "").replace("feet", "").replace(",", "").strip()
    if not s: return np.nan
    try: return float(s)
    except:
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group(0)) if m else np.nan

def parse_currency(v):
    if pd.isna(v): return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)): return float(v)
    s = clean_text(v).replace("php", "").replace("₱", "").replace(",", "").strip()
    try: return float(s)
    except: return np.nan

def parse_month(v):
    if pd.isna(v): return np.nan
    if isinstance(v, (int, np.integer)): return int(v) if 1 <= int(v) <= 12 else np.nan
    s = clean_text(v).lower()
    if s in MONTH_MAP: return MONTH_MAP[s]
    try:
        n = int(float(s))
        return n if 1 <= n <= 12 else np.nan
    except: return np.nan

def severity_label(row):
    w, f, d = row["water_level_num"], row["families_num"], row["damage_total_num"]
    high_count = 0
    if pd.notna(w) and w > 4.0: high_count += 1
    if pd.notna(f) and f > 200: high_count += 1
    if pd.notna(d) and d > 500000: high_count += 1
    if high_count >= 2: return "High"
    if (pd.isna(w) or w <= 2.0) and (pd.isna(f) or f < 50) and (pd.isna(d) or d < 100000): return "Low"
    return "Moderate"

@st.cache_data(show_spinner=False)
def load_and_clean(file_path):
    df = pd.read_excel(file_path, sheet_name=CONFIG["sheet_index"], engine='openpyxl')
    df.columns = [clean_text(c) for c in df.columns]
    df = df.dropna(how="all")
    for c in df.columns: df[c] = df[c].apply(clean_text)
    
    bc, cc, wc, fc, ic, ac = CONFIG["barangay_col"], CONFIG["cause_col"], CONFIG["water_col"], CONFIG["families_col"], CONFIG["infra_col"], CONFIG["agri_col"]
    
    df[bc] = df[bc].apply(lambda x: clean_text(x).title() if pd.notna(x) else np.nan)
    df[cc] = df[cc].apply(lambda x: clean_text(x).replace("Tropica Storm", "Tropical Storm") if pd.notna(x) else np.nan)
    
    df["water_level_num"] = df[wc].apply(parse_number)
    df["families_num"] = df[fc].apply(parse_number)
    df["damage_infra_num"] = df[ic].apply(parse_currency)
    df["damage_agri_num"] = df[ac].apply(parse_currency)
    df["damage_total_num"] = df[["damage_infra_num", "damage_agri_num"]].sum(axis=1, min_count=1)
    df["month_num"] = df[CONFIG["month_col"]].apply(parse_month)
    df["year_num"] = pd.to_numeric(df[CONFIG["year_col"]], errors='coerce')
    
    df = df.dropna(subset=[bc, cc])
    df["severity"] = df.apply(severity_label, axis=1)
    return df

try:
    df = load_and_clean(CONFIG["file_path"])
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

st.sidebar.title("🌊 Flood Dashboard")
st.sidebar.metric("Total Flood Records", f"{len(df):,}")
st.sidebar.metric("Affected Barangays", df[CONFIG["barangay_col"]].nunique())
st.sidebar.metric("Families Affected", f"{int(df['families_num'].sum(skipna=True)):,}")
st.sidebar.metric("Total Damage (PHP)", f"₱{int(df['damage_total_num'].sum(skipna=True)):,}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Seasonality", "K-Means", "Decision Tree", "Rules & Support"])

with tab1:
    st.header("Flood Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Records by Barangay")
        st.bar_chart(df[CONFIG["barangay_col"]].value_counts())
    with col2:
        st.subheader("Severity Distribution")
        st.bar_chart(df["severity"].value_counts())

with tab2:
    st.header("Seasonality & Weather")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Flood Frequency by Month")
        mcount = df["month_num"].dropna().astype(int).value_counts().sort_index()
        mcount.index = [pd.Timestamp(2024, m, 1).strftime("%B") for m in mcount.index]
        st.bar_chart(mcount)
    with col2:
        st.subheader("Weather System Frequency")
        st.bar_chart(df[CONFIG["cause_col"]].value_counts())

with tab3:
    st.header("K-Means Clustering")
    try:
        cluster_cols = ["water_level_num", "families_num", "damage_infra_num", "damage_agri_num"]
        cdf = df.dropna(subset=cluster_cols).copy()
        
        X = cdf[cluster_cols].values
        X_scaled = StandardScaler().fit_transform(X)
        
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        cdf["cluster"] = km.fit_predict(X_scaled)
        
        cluster_means = cdf.groupby("cluster")["water_level_num"].mean().sort_values()
        mapping = {cluster_means.index[0]: "Low Risk", cluster_means.index[1]: "Moderate Risk", cluster_means.index[2]: "High Risk"}
        cdf["Cluster Profile"] = cdf["cluster"].map(mapping)
        
        prof = cdf.groupby("Cluster Profile").agg(
            Events=("Cluster Profile", "size"),
            Avg_Water_Level=("water_level_num", "mean"),
            Avg_Damage=("damage_total_num", "mean")
        ).round(2).reset_index()
        st.dataframe(prof, hide_index=True)
        
        st.bar_chart(cdf["Cluster Profile"].value_counts())
    except Exception as e:
        st.error(f"Clustering error: {e}")

with tab4:
    st.header("Decision Tree Classification")
    try:
        model_cols = ["water_level_num", "families_num", "damage_infra_num", "damage_agri_num"]
        mdf = df.dropna(subset=model_cols).copy()
        
        Xm = np.array(mdf[model_cols].values, dtype=float)
        ym = np.array(mdf["severity"].values, dtype=str)
        
        X_train, X_test, y_train, y_test = train_test_split(Xm, ym, test_size=0.3, random_state=42)
        dt = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
        col2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.2f}")
        col3.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted', zero_division=0):.2f}")
        
        imp = pd.Series(dt.feature_importances_, index=model_cols)
        st.bar_chart(imp)
    except Exception as e:
        st.error(f"Decision Tree Error: {e}")

with tab5:
    st.header("Association Rules & Decision Support")
    
    try:
        # Custom lightweight Apriori to prevent cloud crashes
        adf = df.dropna(subset=["severity"]).copy().reset_index(drop=True)
        transactions = []
        for _, row in adf.iterrows():
            t = [f"Brgy_{row[CONFIG['barangay_col']]}", f"Severity_{row['severity']}", f"Cause_{row[CONFIG['cause_col']]}"]
            transactions.append(t)
            
        rules = []
        n = len(transactions)
        all_items = set(item for t in transactions for item in t)
        
        for item1, item2 in combinations(all_items, 2):
            count_both = sum(1 for t in transactions if item1 in t and item2 in t)
            count_1 = sum(1 for t in transactions if item1 in t)
            count_2 = sum(1 for t in transactions if item2 in t)
            
            if count_both >= 2 and count_1 > 0 and count_2 > 0:
                support = count_both / n
                conf = count_both / count_1
                lift = conf / (count_2 / n)
                
                if conf > 0.1 and lift > 1.0:
                    rules.append({"If (Antecedent)": item1, "Then (Consequent)": item2, "Support": round(support, 3), "Confidence": round(conf, 3), "Lift": round(lift, 2)})
                    
        rules.sort(key=lambda x: x["Lift"], reverse=True)
        
        if rules:
            st.dataframe(pd.DataFrame(rules[:15]), hide_index=True)
        else:
            st.info("No strong rules found.")
            
        st.divider()
        st.subheader("Dynamic Recommendations (Obj 3)")
        high_brgy = df[df['severity']=='High'][CONFIG["barangay_col"]].value_counts()
        top_high_brgy = high_brgy.index[0] if not high_brgy.empty else df[CONFIG["barangay_col"]].value_counts().index[0]
        top_month_idx = df['month_num'].dropna().value_counts()
        top_month = pd.Timestamp(2024, int(top_month_idx.index[0]), 1).strftime('%B') if not top_month_idx.empty else "January"
        
        st.success(f"""
        **Target Barangay:** **{top_high_brgy}** (Highest number of High-Severity floods)
        **Target Month:** **{top_month}** (Peak flood frequency)
        """)
    except Exception as e:
        st.error(f"Rule mining error: {e}")
