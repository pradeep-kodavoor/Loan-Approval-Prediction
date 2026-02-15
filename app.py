"""
Loan Approval Prediction — Interactive ML Dashboard
=====================================================
Streamlit application for comparing classification models
trained on the Loan Approval Prediction dataset.

Author: Pradeep Kodavoor
Course: M.Tech AIML - Machine Learning Assignment 2
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────
# APP CONFIGURATION
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
ARTIFACTS_DIR = "model"

CLASSIFIER_MAP = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
    "XGBoost (Ensemble)": "xgboost_ensemble.pkl",
}

STANDARDIZED_INPUT = {"Logistic Regression", "KNN"}


# ──────────────────────────────────────────────
# CACHED LOADERS
# ──────────────────────────────────────────────
@st.cache_resource
def fetch_model(classifier_name):
    pkl_file = CLASSIFIER_MAP[classifier_name]
    return joblib.load(os.path.join(ARTIFACTS_DIR, pkl_file))

@st.cache_resource
def fetch_scaler():
    return joblib.load(os.path.join(ARTIFACTS_DIR, "std_scaler.pkl"))

@st.cache_resource
def fetch_feature_list():
    return joblib.load(os.path.join(ARTIFACTS_DIR, "feature_columns.pkl"))


def calculate_all_metrics(truth, preds, probs):
    return {
        "Accuracy": accuracy_score(truth, preds),
        "AUC Score": roc_auc_score(truth, probs),
        "Precision": precision_score(truth, preds),
        "Recall": recall_score(truth, preds),
        "F1 Score": f1_score(truth, preds),
        "MCC": matthews_corrcoef(truth, preds),
    }


def render_confusion_matrix(truth, preds, title):
    cm = confusion_matrix(truth, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['Rejected', 'Approved'],
        yticklabels=['Rejected', 'Approved'],
        linewidths=1, linecolor='white'
    )
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    chosen_classifier = st.selectbox(
        "Choose a Classifier",
        list(CLASSIFIER_MAP.keys()),
        help="Pick one of the 6 trained ML models to evaluate"
    )

    st.divider()

    st.markdown("**Dataset Info**")
    st.markdown("""
    - **Source:** Kaggle  
    - **Task:** Binary Classification  
    - **Samples:** 4,269  
    - **Features:** 13  
    - **Target:** Loan Status (Approved/Rejected)
    """)

    st.divider()

    st.markdown("**Features**")
    feature_info = {
        "no_of_dependents": "Number of dependents",
        "education": "Graduate or Not Graduate",
        "self_employed": "Self-employed status",
        "income_annum": "Annual income",
        "loan_amount": "Requested loan amount",
        "loan_term": "Loan term in months",
        "cibil_score": "Credit score (300-900)",
        "residential_assets_value": "Residential asset value",
        "commercial_assets_value": "Commercial asset value",
        "luxury_assets_value": "Luxury asset value",
        "bank_asset_value": "Bank asset value",
    }
    for feat, desc in feature_info.items():
        st.caption(f"**{feat}** — {desc}")


# ──────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────
st.title("🏦 Loan Approval Prediction System")
st.markdown("Compare 6 ML classifiers on the Loan Approval Prediction dataset")

st.divider()

st.subheader("Upload Evaluation Data")
st.caption("Upload a CSV with features + loan_status column. Use `test_data.csv` from training.")

uploaded_csv = st.file_uploader("Select CSV file", type=["csv"])

if uploaded_csv is not None:
    try:
        input_df = pd.read_csv(uploaded_csv)
        st.success(f"Loaded {input_df.shape[0]} samples with {input_df.shape[1]} columns")

        with st.expander("Data Preview"):
            st.dataframe(input_df.head(10), use_container_width=True)

        if 'loan_status' not in input_df.columns:
            st.error("CSV must include a 'loan_status' column (0 = Rejected, 1 = Approved)")
            st.stop()

        feature_list = fetch_feature_list()
        scaler = fetch_scaler()

        missing_cols = [c for c in feature_list if c not in input_df.columns]
        if missing_cols:
            st.error(f"Missing columns in uploaded file: {missing_cols}")
            st.stop()

        X_input = input_df[feature_list]
        y_actual = input_df['loan_status']

        # Model Inference
        st.subheader(f"Results: {chosen_classifier}")

        classifier = fetch_model(chosen_classifier)

        if chosen_classifier in STANDARDIZED_INPUT:
            X_processed = scaler.transform(X_input)
        else:
            X_processed = X_input

        y_predicted = classifier.predict(X_processed)
        y_proba = classifier.predict_proba(X_processed)[:, 1]

        # Metrics
        eval_metrics = calculate_all_metrics(y_actual, y_predicted, y_proba)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Accuracy", f"{eval_metrics['Accuracy']:.4f}")
        c2.metric("AUC", f"{eval_metrics['AUC Score']:.4f}")
        c3.metric("Precision", f"{eval_metrics['Precision']:.4f}")
        c4.metric("Recall", f"{eval_metrics['Recall']:.4f}")
        c5.metric("F1 Score", f"{eval_metrics['F1 Score']:.4f}")
        c6.metric("MCC", f"{eval_metrics['MCC']:.4f}")

        st.divider()

        # Confusion Matrix & Report
        left_panel, right_panel = st.columns([1, 1])

        with left_panel:
            st.subheader("Confusion Matrix")
            cm_fig = render_confusion_matrix(y_actual, y_predicted, chosen_classifier)
            st.pyplot(cm_fig)

        with right_panel:
            st.subheader("Classification Report")
            report_dict = classification_report(
                y_actual, y_predicted,
                target_names=['Rejected', 'Approved'],
                output_dict=True
            )
            report_table = pd.DataFrame(report_dict).transpose()
            st.dataframe(
                report_table.style.format("{:.4f}"),
                use_container_width=True
            )

        st.divider()

        # All Models Comparison
        st.subheader("All Classifiers — Comparison")

        summary_path = os.path.join(ARTIFACTS_DIR, "performance_summary.csv")
        if os.path.exists(summary_path):
            comparison_df = pd.read_csv(summary_path)
        else:
            comparison_rows = []
            for cname in CLASSIFIER_MAP:
                clf = fetch_model(cname)
                if cname in STANDARDIZED_INPUT:
                    xp = scaler.transform(X_input)
                else:
                    xp = X_input
                yp = clf.predict(xp)
                ypr = clf.predict_proba(xp)[:, 1]
                row = calculate_all_metrics(y_actual, yp, ypr)
                row["Model"] = cname
                comparison_rows.append(row)
            comparison_df = pd.DataFrame(comparison_rows)
            cols = ["Model"] + [c for c in comparison_df.columns if c != "Model"]
            comparison_df = comparison_df[cols]

        metric_cols = [c for c in comparison_df.columns if c != "Model"]
        st.dataframe(
            comparison_df.style.highlight_max(
                subset=metric_cols, color='lightgreen'
            ).format({c: "{:.4f}" for c in metric_cols}),
            use_container_width=True
        )

        st.divider()

        # Predictions Preview
        st.subheader("Sample Predictions")

        output_df = input_df.copy()
        output_df['Prediction'] = y_predicted
        output_df['Probability'] = np.round(y_proba, 4)
        output_df['Match'] = np.where(
            output_df['loan_status'] == output_df['Prediction'], '✅ Correct', '❌ Wrong'
        )

        correct_count = (y_actual == y_predicted).sum()
        total_count = len(y_actual)
        st.caption(f"{correct_count}/{total_count} classified correctly ({correct_count/total_count*100:.1f}%)")
        st.dataframe(output_df.head(15), use_container_width=True)

        # Feature Importance
        if chosen_classifier in ["Decision Tree", "Random Forest (Ensemble)", "XGBoost (Ensemble)"]:
            st.divider()
            st.subheader(f"Feature Importance — {chosen_classifier}")

            importances = classifier.feature_importances_
            fi_df = pd.DataFrame({
                'Feature': feature_list,
                'Importance': importances
            }).sort_values('Importance', ascending=True)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(fi_df['Feature'], fi_df['Importance'], color='steelblue')
            ax.set_xlabel('Importance')
            ax.set_title(f'{chosen_classifier} — Feature Importance')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

    except Exception as err:
        st.error(f"Something went wrong: {str(err)}")
        st.exception(err)

else:
    st.info("Upload a test CSV file to begin evaluation")

    st.markdown("**Expected CSV Columns:**")
    st.code(
        "no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, "
        "cibil_score, residential_assets_value, commercial_assets_value, luxury_assets_value, "
        "bank_asset_value, loan_status",
        language=None
    )

    st.markdown("**Sample Data:**")
    demo = pd.DataFrame({
        'no_of_dependents': [2, 0, 3, 1, 4],
        'education': [1, 1, 0, 1, 0],
        'self_employed': [0, 0, 1, 0, 1],
        'income_annum': [4900000, 9200000, 3000000, 7100000, 2500000],
        'loan_amount': [12200000, 16300000, 8000000, 14400000, 5600000],
        'loan_term': [8, 12, 6, 10, 4],
        'cibil_score': [778, 614, 442, 712, 350],
        'residential_assets_value': [2400000, 7800000, 1200000, 5100000, 900000],
        'commercial_assets_value': [1700000, 5400000, 0, 3200000, 0],
        'luxury_assets_value': [2200000, 8900000, 1500000, 6300000, 1100000],
        'bank_asset_value': [800000, 3100000, 400000, 2100000, 200000],
        'loan_status': [1, 0, 0, 1, 0],
    })
    st.dataframe(demo, use_container_width=True)

    st.caption("Tip: Use the test_data.csv file generated during model training.")
