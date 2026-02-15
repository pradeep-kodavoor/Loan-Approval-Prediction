"""
Loan Approval Prediction - Training Pipeline
==============================================
This script trains 6 ML classifiers on the Loan Approval Prediction dataset,
evaluates performance using multiple metrics, and saves trained models.

Author: Pradeep Kodavoor
"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
DATASET_FILE = "loan_approval_dataset.csv"
SAVED_MODELS_DIR = "model"
TEST_SPLIT_RATIO = 0.2
RANDOM_SEED = 42
KNN_NEIGHBORS = 7
RF_TREES = 150
XGB_ROUNDS = 120

os.makedirs(SAVED_MODELS_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────
def find_dataset(filename):
    search_paths = [filename, f"model/{filename}", f"../{filename}", f"data/{filename}"]
    for p in search_paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Dataset '{filename}' not found. Please place it in the project root.")


def evaluate_classifier(actual, predicted, prob_scores):
    return {
        "Accuracy": round(accuracy_score(actual, predicted), 4),
        "AUC": round(roc_auc_score(actual, prob_scores), 4),
        "Precision": round(precision_score(actual, predicted), 4),
        "Recall": round(recall_score(actual, predicted), 4),
        "F1 Score": round(f1_score(actual, predicted), 4),
        "MCC": round(matthews_corrcoef(actual, predicted), 4),
    }


def save_artifact(obj, name):
    filepath = os.path.join(SAVED_MODELS_DIR, f"{name}.pkl")
    joblib.dump(obj, filepath)
    print(f"    → Saved: {filepath}")
    return filepath


# ──────────────────────────────────────────────
# STEP 1: LOAD & EXPLORE DATASET
# ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 1: Loading & Exploring the Dataset")
print("=" * 55)

dataset_path = find_dataset(DATASET_FILE)
loan_df = pd.read_csv(dataset_path)

# Clean column names (remove leading/trailing spaces)
loan_df.columns = loan_df.columns.str.strip()

print(f"\n  Source file: {dataset_path}")
print(f"  Shape: {loan_df.shape[0]} rows × {loan_df.shape[1]} columns")
print(f"\n  Columns: {list(loan_df.columns)}")
print(f"\n  First 5 rows:")
print(loan_df.head().to_string())
print(f"\n  Data Types:\n{loan_df.dtypes.to_string()}")
print(f"\n  Missing values: {loan_df.isnull().sum().sum()}")

# Clean the target column
loan_df['loan_status'] = loan_df['loan_status'].str.strip()

print(f"\n  Target distribution:")
print(loan_df['loan_status'].value_counts().to_string())


# ──────────────────────────────────────────────
# STEP 2: DATA PREPROCESSING
# ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 2: Data Preprocessing")
print("=" * 55)

# Encode categorical columns
label_encoders = {}

# education: Graduate=1, Not Graduate=0
if loan_df['education'].dtype == 'object':
    loan_df['education'] = loan_df['education'].str.strip()
    le_edu = LabelEncoder()
    loan_df['education'] = le_edu.fit_transform(loan_df['education'])
    label_encoders['education'] = le_edu
    print(f"  → Encoded 'education': {dict(zip(le_edu.classes_, le_edu.transform(le_edu.classes_)))}")

# self_employed: Yes=1, No=0
if loan_df['self_employed'].dtype == 'object':
    loan_df['self_employed'] = loan_df['self_employed'].str.strip()
    le_emp = LabelEncoder()
    loan_df['self_employed'] = le_emp.fit_transform(loan_df['self_employed'])
    label_encoders['self_employed'] = le_emp
    print(f"  → Encoded 'self_employed': {dict(zip(le_emp.classes_, le_emp.transform(le_emp.classes_)))}")

# Encode target: Approved=1, Rejected=0
le_target = LabelEncoder()
loan_df['loan_status'] = le_target.fit_transform(loan_df['loan_status'])
label_encoders['loan_status'] = le_target
print(f"  → Encoded 'loan_status': {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

# Save label encoders
save_artifact(label_encoders, "label_encoders")

print(f"\n  Processed DataFrame shape: {loan_df.shape}")
print(f"  Columns: {list(loan_df.columns)}")
print(f"\n  Statistical Summary:")
print(loan_df.describe().round(2).to_string())


# ──────────────────────────────────────────────
# STEP 3: EXPLORATORY DATA ANALYSIS
# ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 3: Exploratory Data Analysis")
print("=" * 55)

# 3a. Correlation heatmap
plt.figure(figsize=(12, 8))
correlation = loan_df.corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SAVED_MODELS_DIR, "eda_correlation_heatmap.png"), dpi=150)
plt.close()
print("  → Correlation heatmap saved")

# 3b. Target distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
loan_df['loan_status'].value_counts().plot(kind='bar', ax=axes[0],
    color=['#e74c3c', '#2ecc71'], edgecolor='black')
axes[0].set_title('Loan Approval Distribution', fontweight='bold')
axes[0].set_xticklabels(['Rejected', 'Approved'], rotation=0)
axes[0].set_ylabel('Count')

# CIBIL score distribution by loan status
loan_df[loan_df['loan_status'] == 0]['cibil_score'].hist(ax=axes[1], alpha=0.6,
    bins=30, color='#e74c3c', label='Rejected')
loan_df[loan_df['loan_status'] == 1]['cibil_score'].hist(ax=axes[1], alpha=0.6,
    bins=30, color='#2ecc71', label='Approved')
axes[1].set_title('CIBIL Score Distribution by Loan Status', fontweight='bold')
axes[1].set_xlabel('CIBIL Score')
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVED_MODELS_DIR, "eda_distributions.png"), dpi=150)
plt.close()
print("  → Distribution plots saved")

# 3c. Key feature boxplots
key_features = ['income_annum', 'loan_amount', 'loan_term', 'cibil_score']
fig, axes = plt.subplots(1, len(key_features), figsize=(18, 4))
for i, feat in enumerate(key_features):
    loan_df.boxplot(column=feat, by='loan_status', ax=axes[i])
    axes[i].set_title(feat, fontweight='bold')
    axes[i].set_xlabel('Loan Status')
plt.suptitle('Key Features by Loan Status', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SAVED_MODELS_DIR, "eda_boxplots.png"), dpi=150)
plt.close()
print("  → Boxplots saved")


# ──────────────────────────────────────────────
# STEP 4: SPLIT & SCALE DATA
# ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 4: Splitting & Scaling Data")
print("=" * 55)

feature_cols = [col for col in loan_df.columns if col != 'loan_status']
X_data = loan_df[feature_cols]
y_label = loan_df['loan_status']

X_train, X_eval, y_train, y_eval = train_test_split(
    X_data, y_label,
    test_size=TEST_SPLIT_RATIO,
    random_state=RANDOM_SEED,
    stratify=y_label
)

print(f"  Training samples: {X_train.shape[0]}")
print(f"  Evaluation samples: {X_eval.shape[0]}")
print(f"  Features used: {len(feature_cols)} → {feature_cols}")

std_scaler = StandardScaler()
X_train_norm = std_scaler.fit_transform(X_train)
X_eval_norm = std_scaler.transform(X_eval)

save_artifact(std_scaler, "std_scaler")
save_artifact(feature_cols, "feature_columns")

# Export evaluation set for Streamlit app
eval_export = pd.DataFrame(X_eval, columns=feature_cols)
eval_export['loan_status'] = y_eval.values
eval_export.to_csv("test_data.csv", index=False)
print("  → Evaluation data exported to test_data.csv")


# ──────────────────────────────────────────────
# STEP 5: TRAIN & EVALUATE ALL CLASSIFIERS
# ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 5: Training & Evaluating Classifiers")
print("=" * 55)

classifiers = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, solver='lbfgs', random_state=RANDOM_SEED
    ),
    "Decision Tree": DecisionTreeClassifier(
        criterion='gini', max_depth=None, random_state=RANDOM_SEED
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=KNN_NEIGHBORS, weights='uniform', metric='minkowski'
    ),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=RF_TREES, criterion='gini',
        max_features='sqrt', random_state=RANDOM_SEED
    ),
    "XGBoost (Ensemble)": XGBClassifier(
        n_estimators=XGB_ROUNDS, learning_rate=0.1,
        max_depth=6, random_state=RANDOM_SEED,
        use_label_encoder=False, eval_metric='logloss'
    ),
}

needs_scaling = {"Logistic Regression", "KNN"}

performance_log = []

for clf_name, clf_obj in classifiers.items():
    print(f"\n  ╔══ {clf_name} ══╗")

    if clf_name in needs_scaling:
        train_x, eval_x = X_train_norm, X_eval_norm
    else:
        train_x, eval_x = X_train, X_eval

    clf_obj.fit(train_x, y_train)

    predictions = clf_obj.predict(eval_x)
    probability = clf_obj.predict_proba(eval_x)[:, 1]

    scores = evaluate_classifier(y_eval, predictions, probability)
    scores["Model"] = clf_name
    performance_log.append(scores)

    for metric_name, metric_val in scores.items():
        if metric_name != "Model":
            print(f"    {metric_name:>12}: {metric_val}")

    cm = confusion_matrix(y_eval, predictions)
    print(f"\n    Confusion Matrix:")
    print(f"      TN={cm[0][0]:>4}  FP={cm[0][1]:>4}")
    print(f"      FN={cm[1][0]:>4}  TP={cm[1][1]:>4}")

    print(f"\n    Classification Report:")
    print(classification_report(y_eval, predictions))

    safe_filename = clf_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    save_artifact(clf_obj, safe_filename)


# ──────────────────────────────────────────────
# STEP 6: RESULTS SUMMARY
# ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 6: Performance Summary")
print("=" * 55)

summary_df = pd.DataFrame(performance_log)
col_order = ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
summary_df = summary_df[col_order]
print(f"\n{summary_df.to_string(index=False)}")

summary_df.to_csv(os.path.join(SAVED_MODELS_DIR, "performance_summary.csv"), index=False)
print(f"\n  → Summary saved to {SAVED_MODELS_DIR}/performance_summary.csv")

best_row = summary_df.loc[summary_df['Accuracy'].idxmax()]
print(f"\n  Best Model: {best_row['Model']} (Accuracy: {best_row['Accuracy']})")


# ──────────────────────────────────────────────
# STEP 7: VISUALIZATIONS
# ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 7: Generating Visualizations")
print("=" * 55)

# 7a. Confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

for idx, (clf_name, clf_obj) in enumerate(classifiers.items()):
    if clf_name in needs_scaling:
        preds = clf_obj.predict(X_eval_norm)
    else:
        preds = clf_obj.predict(X_eval)

    cm = confusion_matrix(y_eval, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=axes[idx],
                xticklabels=['Rejected', 'Approved'],
                yticklabels=['Rejected', 'Approved'],
                cbar_kws={'shrink': 0.8})
    axes[idx].set_title(clf_name, fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')

plt.suptitle('Confusion Matrices — All Classifiers', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVED_MODELS_DIR, "all_confusion_matrices.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  → Confusion matrices saved")

# 7b. Metric comparison chart
fig, ax = plt.subplots(figsize=(14, 6))
metric_names = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
n_models = len(summary_df)
bar_width = 0.13
positions = np.arange(n_models)

colors = ['#2c3e50', '#2980b9', '#27ae60', '#f39c12', '#e74c3c', '#8e44ad']
for i, metric in enumerate(metric_names):
    ax.bar(positions + i * bar_width, summary_df[metric],
           bar_width, label=metric, color=colors[i], edgecolor='white')

ax.set_xlabel('Classifier', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Classifier Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(positions + bar_width * 2.5)
ax.set_xticklabels(summary_df['Model'], rotation=20, ha='right', fontsize=9)
ax.legend(loc='lower right', fontsize=9)
ax.set_ylim(0, 1.12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVED_MODELS_DIR, "metric_comparison_chart.png"), dpi=150)
plt.close()
print("  → Comparison chart saved")

# 7c. Feature importance from Random Forest
rf_model = classifiers["Random Forest (Ensemble)"]
importances = rf_model.feature_importances_
feat_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feat_importance_df['Feature'], feat_importance_df['Importance'],
        color='#3498db', edgecolor='white')
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_title('Random Forest — Feature Importance', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVED_MODELS_DIR, "feature_importance.png"), dpi=150)
plt.close()
print("  → Feature importance plot saved")


print("\n" + "=" * 55)
print("  PIPELINE COMPLETE")
print("  All models and artifacts saved in '{}'".format(SAVED_MODELS_DIR))
print("=" * 55 + "\n")
