"""
visual_app.py
Clinical Trial Data Visualisation - Streamlit App

Run:  python -m streamlit run visual_app.py

Files needed in the SAME folder:
  clinical_trials_raw_patient2trial_conditions.csv   (your dataset)
  viz_info.pkl                                       (optional - for the Model Results tab)
"""

import os
import re
from collections import Counter

import joblib
import nltk
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords

st.set_page_config(page_title="Clinical Trial Visualisation", page_icon="📊", layout="wide")

CSV_FILE = "clinical_trials_raw_patient2trial_conditions.csv"
sns.set_style("whitegrid")

# ═══════════════════════════════════════════════════════════
# LOAD DATA (cached - runs once)
# ═══════════════════════════════════════════════════════════


def age_to_years(value):
    if pd.isna(value):
        return np.nan
    parts = str(value).split()
    try:
        number = float(parts[0])
    except ValueError:
        return np.nan
    unit = parts[1].lower() if len(parts) > 1 else "years"
    if "month" in unit:
        return number / 12
    if "week" in unit:
        return number / 52
    if "day" in unit:
        return number / 365
    return number


@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    columns = [
        "source_condition_query", "brief_summary", "conditions", "interventions",
        "overall_status", "study_type", "phase", "sex", "minimum_age",
        "maximum_age", "healthy_volunteers", "eligibility_criteria",
    ]
    df = pd.read_csv(CSV_FILE, usecols=columns)
    df["summary_words"] = df["brief_summary"].fillna("").str.split().str.len()
    df["eligibility_words"] = df["eligibility_criteria"].fillna("").str.split().str.len()
    df["min_age_years"] = df["minimum_age"].apply(age_to_years)
    df["max_age_years"] = df["maximum_age"].apply(age_to_years)
    df["phase"] = df["phase"].fillna("Not applicable")
    df["sex"] = df["sex"].fillna("Unknown")
    df["healthy_volunteers"] = df["healthy_volunteers"].astype(str).replace("nan", "Unknown")
    return df.drop(columns=["eligibility_criteria"])


@st.cache_resource
def get_stop_words():
    nltk.download("stopwords", quiet=True)
    return set(stopwords.words("english"))


@st.cache_data(show_spinner="Counting words...")
def top_words_for(disease, n=15):
    data = load_data()
    summaries = data[data["source_condition_query"] == disease]["brief_summary"].dropna()
    stop_words = get_stop_words()
    counter = Counter()
    for text in summaries:
        text = re.sub(r"[^a-z\s]", "", text.lower())
        counter.update(w for w in text.split() if w not in stop_words and len(w) > 2)
    return counter.most_common(n)


def top_items(series, n=15):
    counter = Counter()
    for row in series.dropna():
        for item in row.split(" | "):
            counter[item.strip()] += 1
    return counter.most_common(n)


def bar(labels, values, title, xlabel, color="steelblue", horizontal=True, figsize=(8, 4.5)):
    fig, ax = plt.subplots(figsize=figsize)
    if horizontal:
        ax.barh([str(l) for l in labels], values, color=color)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
    else:
        ax.bar([str(l) for l in labels], values, color=color)
        ax.set_ylabel(xlabel)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# START
# ═══════════════════════════════════════════════════════════

st.title("📊 Clinical Trial Data Visualisation")

if not os.path.exists(CSV_FILE):
    st.error(f"Dataset not found: {CSV_FILE}. Keep the CSV in the same folder as this app.")
    st.stop()

df_all = load_data()

# Sidebar filter
st.sidebar.header("Filter")
all_diseases = sorted(df_all["source_condition_query"].unique())
chosen = st.sidebar.multiselect("Diseases to show", all_diseases, default=all_diseases)
if not chosen:
    st.warning("Select at least one disease in the sidebar.")
    st.stop()
df = df_all[df_all["source_condition_query"].isin(chosen)]
st.sidebar.write(f"Showing **{len(df):,}** trials")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📌 Overview", "🧪 Trial Details", "👥 Age", "📝 Text Analysis", "🎯 Model Results"]
)

# ───────────────────────────────────────────────────────────
# TAB 1 - OVERVIEW
# ───────────────────────────────────────────────────────────
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total trials", f"{len(df):,}")
    c2.metric("Diseases", df["source_condition_query"].nunique())
    c3.metric("Completed", f"{(df['overall_status'] == 'COMPLETED').mean() * 100:.1f}%")
    c4.metric("Interventional", f"{(df['study_type'] == 'INTERVENTIONAL').mean() * 100:.1f}%")

    counts = df["source_condition_query"].value_counts()
    left, right = st.columns(2)
    with left:
        show(bar([c.title() for c in counts.index], counts.values,
                 "Trials per disease", "Number of trials", color="steelblue"))
    with right:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.pie(counts.values, labels=[c.title() for c in counts.index],
               autopct="%1.1f%%", startangle=90)
        ax.set_title("Share of each disease")
        show(fig)

# ───────────────────────────────────────────────────────────
# TAB 2 - TRIAL DETAILS
# ───────────────────────────────────────────────────────────
with tab2:
    options = {
        "Overall status": "overall_status",
        "Study type": "study_type",
        "Phase": "phase",
        "Sex eligibility": "sex",
        "Healthy volunteers": "healthy_volunteers",
    }
    label = st.selectbox("Choose a column", list(options.keys()))
    column = options[label]

    left, right = st.columns(2)
    with left:
        vc = df[column].value_counts()
        show(bar(vc.index, vc.values, f"{label} - all trials", "Number of trials", color="coral"))
    with right:
        pct = pd.crosstab(df["source_condition_query"], df[column], normalize="index") * 100
        fig, ax = plt.subplots(figsize=(8, 4.5))
        pct.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
        ax.set_title(f"{label} by disease (%)")
        ax.set_ylabel("Percentage")
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        show(fig)

# ───────────────────────────────────────────────────────────
# TAB 3 - AGE
# ───────────────────────────────────────────────────────────
with tab3:
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.histplot(df["min_age_years"].dropna(), bins=30, color="purple", ax=ax)
        ax.set_title("Minimum age (years)")
        show(fig)
    with right:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.histplot(df["max_age_years"].dropna(), bins=30, color="orange", ax=ax)
        ax.set_title("Maximum age (years) - only trials that set one")
        show(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.boxplot(x="source_condition_query", y="min_age_years", data=df, ax=ax)
    ax.set_title("Minimum age by disease")
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    show(fig)

# ───────────────────────────────────────────────────────────
# TAB 4 - TEXT ANALYSIS
# ───────────────────────────────────────────────────────────
with tab4:
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.boxplot(x="source_condition_query", y="summary_words", data=df,
                    showfliers=False, ax=ax)
        ax.set_title("Brief summary length by disease (words)")
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        show(fig)
    with right:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.histplot(df["eligibility_words"].clip(upper=1500), bins=40, color="darkgreen", ax=ax)
        ax.set_title("Eligibility criteria length (words, capped at 1500)")
        show(fig)

    st.subheader("Most frequent words in the summaries")
    pick = st.selectbox("Choose a disease", chosen)
    words = top_words_for(pick)
    show(bar([w for w, _ in words], [c for _, c in words],
             f"Top words in {pick.title()} trials", "Frequency", color="darkcyan"))

    left, right = st.columns(2)
    with left:
        top_c = top_items(df["conditions"])
        show(bar([c for c, _ in top_c], [n for _, n in top_c],
                 "Top 15 conditions", "Number of trials", color="teal"))
    with right:
        top_i = top_items(df["interventions"])
        show(bar([c for c, _ in top_i], [n for _, n in top_i],
                 "Top 15 interventions", "Number of trials", color="crimson"))

# ───────────────────────────────────────────────────────────
# TAB 5 - MODEL RESULTS
# ───────────────────────────────────────────────────────────
with tab5:
    if not os.path.exists("viz_info.pkl"):
        st.info(
            "Model results are not saved yet. Run the save cell in your notebook "
            "to create viz_info.pkl, then put it in this folder."
        )
    else:
        info = joblib.load("viz_info.pkl")
        classes = [c.title() for c in info["classes"]]
        model_names = list(info["models"].keys())

        acc_cols = st.columns(len(model_names))
        for col, name in zip(acc_cols, model_names):
            col.metric(f"{name} accuracy", f"{info['models'][name]['accuracy'] * 100:.2f}%")

        model_choice = st.selectbox("Show confusion matrix for", model_names)
        view = st.radio("Show as", ["Counts", "Percent of each actual disease"], horizontal=True)

        cm = np.array(info["models"][model_choice]["cm"], dtype=float)
        if view == "Counts":
            data, fmt, cmap = cm, ".0f", "Blues"
        else:
            data, fmt, cmap = cm / cm.sum(axis=1, keepdims=True) * 100, ".1f", "Greens"

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(pd.DataFrame(data, index=classes, columns=classes),
                    annot=True, fmt=fmt, cmap=cmap, linewidths=0.5, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion matrix - {model_choice}")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        show(fig)

        st.subheader("Correct % for each disease")
        table = pd.DataFrame({"Disease": classes})
        for name in model_names:
            m = np.array(info["models"][name]["cm"], dtype=float)
            table[name] = (np.diag(m) / m.sum(axis=1) * 100).round(1)
        st.dataframe(table, hide_index=True)
