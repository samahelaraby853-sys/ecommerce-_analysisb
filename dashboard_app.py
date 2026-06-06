"""
Lab 10 — Streamlit Dashboard
Data Analysis with Python · GUC CCE

Interactive dashboard for Titanic + E-Commerce datasets.

Run:
    streamlit run dashboard_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score, mean_squared_error,
                             accuracy_score, classification_report,
                             confusion_matrix, roc_curve, roc_auc_score,
                             precision_score, recall_score, f1_score)
import os
import io

sns.set_theme(style="whitegrid", font_scale=1.0)

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")
st.title("Data Analysis Dashboard")
st.caption("Lab 10 · Data Analysis with Python · GUC CCE")


# ── Load data ────────────────────────────────────────────────────
@st.cache_data
def load_titanic():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df = df.drop(columns=["Cabin"]).drop_duplicates()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["Sex_encoded"] = df["Sex"].map({"male": 0, "female": 1})
    return df


@st.cache_data
def load_ecommerce():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "Ecommerce_Sales_Analysis.csv")
    if not os.path.exists(path):
        path = os.path.join(base, "..", "..", "Assignments", "Ecommerce_Sales_Analysis.csv")
    df = pd.read_csv(path)
    df["is_profitable"] = (df["Profit"] > 0).astype(int)
    return df


# ── Sidebar ──────────────────────────────────────────────────────
st.sidebar.header("Settings")
dataset_name = st.sidebar.selectbox("Choose Dataset", ["E-Commerce Sales", "Titanic"])

if dataset_name == "Titanic":
    df = load_titanic()
    numeric_cols = ["Age", "Fare", "Pclass", "Survived", "FamilySize", "Sex_encoded"]
    target_reg = "Fare"
    target_cls = "Survived"
    features_reg = ["Age", "Pclass", "FamilySize", "Sex_encoded"]
    features_cls = ["Age", "Pclass", "FamilySize", "Sex_encoded"]
    cls_labels = ["Died", "Survived"]
else:
    df = load_ecommerce()
    cat_cols = ["Category", "Segment", "Region"]
    df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    numeric_cols = ["Sales", "Quantity", "Discount", "Profit"]
    target_reg = "Profit"
    target_cls = "is_profitable"
    features_reg = ["Sales", "Quantity", "Discount",
                    "Category_Office Supplies", "Category_Technology",
                    "Segment_Corporate", "Segment_Home Office",
                    "Region_East", "Region_South", "Region_West"]
    features_cls = features_reg
    cls_labels = ["Not Profitable", "Profitable"]

st.sidebar.markdown("---")
st.sidebar.info(f"**{dataset_name}** — {len(df):,} rows, {len(df.columns)} columns")


# ── Tabs ─────────────────────────────────────────────────────────
tab_eda, tab_models, tab_report = st.tabs(["EDA Explorer", "Prediction Models", "Report"])


# ═══════════════════════════════════════════════════════════════
#  TAB 1 — EDA
# ═══════════════════════════════════════════════════════════════
with tab_eda:
    st.header(f"Exploratory Analysis — {dataset_name}")

    # KPI row
    if dataset_name == "Titanic":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Passengers", f"{len(df):,}")
        c2.metric("Survival Rate", f"{df['Survived'].mean():.1%}")
        c3.metric("Mean Fare", f"£{df['Fare'].mean():.2f}")
        c4.metric("Mean Age", f"{df['Age'].mean():.1f} yrs")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transactions", f"{len(df):,}")
        c2.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
        c3.metric("Total Profit", f"${df['Profit'].sum():,.0f}")
        c4.metric("Profitable %", f"{df['is_profitable'].mean():.1%}")

    st.markdown("---")

    # Descriptive stats
    with st.expander("Descriptive Statistics"):
        st.dataframe(df[numeric_cols].describe().round(2))

    # Charts
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Distribution")
        dist_col = st.selectbox("Select column", numeric_cols, key="dist_col")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.hist(df[dist_col], bins=40, color="steelblue", edgecolor="white", alpha=0.8)
        ax1.set_title(f"{dist_col} Distribution", fontweight="bold")
        ax1.set_xlabel(dist_col)
        ax1.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with col_right:
        st.subheader("Scatter Plot")
        scatter_x = st.selectbox("X axis", numeric_cols, index=0, key="scatter_x")
        scatter_y = st.selectbox("Y axis", numeric_cols,
                                 index=min(1, len(numeric_cols) - 1), key="scatter_y")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.scatter(df[scatter_x], df[scatter_y], alpha=0.3, color="steelblue", s=10)
        ax2.set_title(f"{scatter_y} vs {scatter_x}", fontweight="bold")
        ax2.set_xlabel(scatter_x)
        ax2.set_ylabel(scatter_y)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr = df[numeric_cols].corr().round(2)
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                linewidths=0.5, ax=ax3)
    ax3.set_title("Pearson Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # Raw data
    with st.expander("Show raw data"):
        st.dataframe(df.head(100))


# ═══════════════════════════════════════════════════════════════
#  TAB 2 — MODELS
# ═══════════════════════════════════════════════════════════════
with tab_models:
    st.header(f"Prediction Models — {dataset_name}")

    if dataset_name == "Titanic":
        X = df[features_reg]
    else:
        X = df_enc[features_reg]

    y_reg = df[target_reg]
    y_cls = df[target_cls]

    X_temp, X_test, y_reg_temp, y_reg_test = train_test_split(X, y_reg, test_size=0.20, random_state=42)
    X_train, X_val, y_reg_train, y_reg_val = train_test_split(X_temp, y_reg_temp, test_size=0.25, random_state=42)
    y_cls_train = y_cls.loc[X_train.index]
    y_cls_val = y_cls.loc[X_val.index]
    y_cls_test = y_cls.loc[X_test.index]

    model_col, result_col = st.columns([1, 1])

    # --- Linear Regression ---
    with model_col:
        st.subheader(f"Linear Regression → {target_reg}")
        model_lr = LinearRegression().fit(X_train, y_reg_train)

        lr_rows = []
        for name, Xs, ys in [("Train", X_train, y_reg_train),
                              ("Val", X_val, y_reg_val),
                              ("Test", X_test, y_reg_test)]:
            pred = model_lr.predict(Xs)
            lr_rows.append({
                "Set": name,
                "R²": round(r2_score(ys, pred), 4),
                "RMSE": round(np.sqrt(mean_squared_error(ys, pred)), 2),
            })
        st.dataframe(pd.DataFrame(lr_rows), hide_index=True)

        with st.expander("Coefficients"):
            coef_df = pd.DataFrame({
                "Feature": features_reg,
                "Coefficient": model_lr.coef_.round(3),
            }).sort_values("Coefficient", key=abs, ascending=False)
            st.dataframe(coef_df, hide_index=True)

    # --- Logistic Regression ---
    with result_col:
        st.subheader(f"Logistic Regression → {target_cls}")
        model_log = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_cls_train)

        log_rows = []
        for name, Xs, ys in [("Train", X_train, y_cls_train),
                              ("Val", X_val, y_cls_val),
                              ("Test", X_test, y_cls_test)]:
            pred = model_log.predict(Xs)
            log_rows.append({
                "Set": name,
                "Accuracy": round(accuracy_score(ys, pred), 3),
                "F1": round(f1_score(ys, pred), 3),
            })
        st.dataframe(pd.DataFrame(log_rows), hide_index=True)

        y_pred_test = model_log.predict(X_test)
        y_proba_test = model_log.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_cls_test, y_proba_test)
        st.metric("Test AUC", f"{auc:.3f}")

    st.markdown("---")

    # Confusion matrix + ROC
    cm_col, roc_col = st.columns(2)

    with cm_col:
        st.subheader("Confusion Matrix (Test Set)")
        cm = confusion_matrix(y_cls_test, y_pred_test)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[f"Pred: {l}" for l in cls_labels],
                    yticklabels=[f"Actual: {l}" for l in cls_labels], ax=ax_cm)
        ax_cm.set_title("Confusion Matrix", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with roc_col:
        st.subheader("ROC Curve (Test Set)")
        fpr, tpr, _ = roc_curve(y_cls_test, y_proba_test)
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        ax_roc.plot(fpr, tpr, color="steelblue", linewidth=2,
                    label=f"AUC = {auc:.3f}")
        ax_roc.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="Random")
        ax_roc.fill_between(fpr, tpr, alpha=0.12, color="steelblue")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curve", fontweight="bold")
        ax_roc.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig_roc)
        plt.close(fig_roc)

    # Threshold slider
    st.markdown("---")
    st.subheader("Threshold Tuning")
    threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)
    y_pred_t = (y_proba_test >= threshold).astype(int)
    tc1, tc2, tc3, tc4 = st.columns(4)
    tc1.metric("Accuracy", f"{accuracy_score(y_cls_test, y_pred_t):.3f}")
    tc2.metric("Precision", f"{precision_score(y_cls_test, y_pred_t, zero_division=0):.3f}")
    tc3.metric("Recall", f"{recall_score(y_cls_test, y_pred_t):.3f}")
    tc4.metric("F1", f"{f1_score(y_cls_test, y_pred_t, zero_division=0):.3f}")


# ═══════════════════════════════════════════════════════════════
#  TAB 3 — REPORT
# ═══════════════════════════════════════════════════════════════
with tab_report:
    st.header("Report & Export")

    st.subheader("Key Findings Summary")
    st.markdown(f"""
    **Dataset:** {dataset_name} ({len(df):,} rows)

    **Linear Regression** — predicting {target_reg}:
    - Test R² = {lr_rows[2]['R²']}
    - Test RMSE = {lr_rows[2]['RMSE']}

    **Logistic Regression** — predicting {target_cls}:
    - Test Accuracy = {log_rows[2]['Accuracy']}
    - Test F1 = {log_rows[2]['F1']}
    - Test AUC = {auc:.3f}
    """)

    st.markdown("---")
    st.subheader("Download Data")

    csv_data = df.to_csv(index=False)
    st.download_button(
        label="Download Full Dataset (CSV)",
        data=csv_data,
        file_name=f"{dataset_name.lower().replace(' ', '_')}_data.csv",
        mime="text/csv",
    )

    results_df = pd.DataFrame(lr_rows + log_rows)
    results_csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Model Results (CSV)",
        data=results_csv,
        file_name=f"{dataset_name.lower().replace(' ', '_')}_results.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.subheader("Generate PDF Report")
    st.info("Run `python generate_pdf_report.py` to create a full PDF report with charts and metrics.")

    st.markdown("---")
    st.caption("Lab 10 · Data Analysis with Python · GUC CCE · Mohamed Medhat")
