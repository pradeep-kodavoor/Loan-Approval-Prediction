# 🏦 Loan Approval Prediction — ML Assignment 2

## Problem Statement

Getting a loan approved is a critical step for individuals and businesses, yet the decision-making process at banks involves evaluating numerous financial and personal factors. Can machine learning automate this and make fair, data-driven loan approval decisions?

This project builds and compares 6 classification models to predict whether a loan application will be approved or rejected based on the applicant's financial profile. The models are evaluated across 6 metrics and deployed via an interactive Streamlit web application.

---

## Dataset Description

| Property | Details |
|----------|---------|
| **Name** | Loan Approval Prediction Dataset |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset) |
| **Classification Type** | Binary (Approved / Rejected) |
| **Total Instances** | 4,269 |
| **Number of Features** | 12 |
| **Target Column** | `loan_status` → Approved or Rejected |
| **Missing Values** | None |

### Features

| # | Feature | What it Represents | Type |
|---|---------|---------------------|------|
| 1 | `loan_id` | Unique identifier for each loan application | Numeric |
| 2 | `no_of_dependents` | Number of dependents of the applicant | Numeric |
| 3 | `education` | Education level (Graduate / Not Graduate) | Categorical |
| 4 | `self_employed` | Whether the applicant is self-employed | Categorical |
| 5 | `income_annum` | Annual income of the applicant | Numeric |
| 6 | `loan_amount` | Loan amount requested | Numeric |
| 7 | `loan_term` | Loan repayment term in months | Numeric |
| 8 | `cibil_score` | Credit score of the applicant (300–900) | Numeric |
| 9 | `residential_assets_value` | Value of residential assets | Numeric |
| 10 | `commercial_assets_value` | Value of commercial assets | Numeric |
| 11 | `luxury_assets_value` | Value of luxury assets | Numeric |
| 12 | `bank_asset_value` | Value of bank assets | Numeric |

### Preprocessing Steps
- Encoded `education` (Graduate=1, Not Graduate=0)
- Encoded `self_employed` (Yes=1, No=0)
- Encoded `loan_status` (Approved=1, Rejected=0)
- Applied StandardScaler for Logistic Regression and KNN

---

## Models Implemented

1. **Logistic Regression** — Linear classifier using sigmoid function for probability estimation
2. **Decision Tree** — Hierarchical rule-based model that splits features at optimal thresholds
3. **K-Nearest Neighbors (KNN)** — Distance-based classifier using k=7 nearest neighbors
4. **Gaussian Naive Bayes** — Probabilistic classifier assuming feature independence
5. **Random Forest (Ensemble)** — Bagging ensemble of 150 decision trees
6. **XGBoost (Ensemble)** — Gradient boosting ensemble with 120 sequential trees

### Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.9215 | 0.9743 | 0.9156 | 0.8731 | 0.8938 | 0.8323 |
| Decision Tree | 0.9766 | 0.9727 | 0.9810 | 0.9567 | 0.9687 | 0.9502 |
| KNN | 0.8923 | 0.9582 | 0.8787 | 0.8297 | 0.8535 | 0.7692 |
| Naive Bayes | 0.7670 | 0.9595 | 0.9366 | 0.4118 | 0.5720 | 0.5142 |
| Random Forest (Ensemble) | 0.9789 | 0.9980 | 0.9841 | 0.9598 | 0.9718 | 0.9552 |
| XGBoost (Ensemble) | 0.9813 | 0.9990 | 0.9873 | 0.9628 | 0.9749 | 0.9602 |

---

## Observations on Model Performance

| ML Model Name | Observation |
|---------------|-------------|
| **Logistic Regression** | Logistic Regression delivers 92.15% accuracy, which is a strong baseline for a linear model. Its precision (0.92) and recall (0.87) are both solid, meaning it correctly approves most deserving applicants while keeping false approvals low. The AUC of 0.97 is particularly impressive — it shows that the model ranks applicants very well even if its hard decision boundary isn't perfect. The slight drop in recall compared to precision suggests it leans conservative, rejecting a few applicants who should have been approved. This makes sense since a linear boundary can't fully capture the non-linear relationships between features like CIBIL score thresholds and income-to-loan ratios. |
| **Decision Tree** | Decision Tree achieves an excellent 97.66% accuracy with nearly balanced precision (0.98) and recall (0.96). It correctly identifies the vast majority of both approved and rejected cases. This is expected because loan approval decisions often follow rule-based logic — for example, "if CIBIL score > 650 and income > X, approve" — which aligns perfectly with how decision trees split data. The MCC of 0.95 confirms strong balanced performance across both classes. One thing to note is its AUC (0.97) is slightly lower than simpler models, which can happen because a single tree produces less smooth probability estimates compared to ensemble methods. |
| **KNN** | KNN comes in at 89.23% accuracy with balanced precision (0.88) and recall (0.83). The AUC of 0.96 indicates it separates approved and rejected applicants well in the feature space. However, it's the second-weakest model here, which is understandable — with 12 features, the distance calculations that KNN relies on become less discriminative (the curse of dimensionality). Applicants who are borderline cases with similar financial profiles to both approved and rejected groups are harder for KNN to classify. Feature scaling via StandardScaler was essential here, as features like income (millions) and loan term (single digits) have vastly different ranges. |
| **Naive Bayes** | Naive Bayes is the clear underperformer at 76.58% accuracy and a very low recall of 0.41 — meaning it misses nearly 60% of applicants who should be approved. Its precision is high (0.94), so when it does approve someone, it's almost always right, but it's far too conservative. The F1 score of 0.57 and MCC of 0.51 reflect this heavy imbalance. The root cause is the independence assumption — Naive Bayes treats each feature as unrelated, but in reality, income, assets, and CIBIL score are strongly correlated. Interestingly, the AUC is still 0.96, meaning the probability rankings are decent even if the decision threshold produces poor classifications. Adjusting the threshold could improve recall significantly. |
| **Random Forest (Ensemble)** | Random Forest achieves 97.89% accuracy with an excellent F1 score (0.97). Its precision of 0.98 means almost zero false approvals, and recall of 0.96 means it catches nearly all genuinely eligible applicants. The AUC of 0.998 is near-perfect, showing outstanding discrimination ability. By aggregating 150 individually trained trees — each seeing a different random slice of data and features — Random Forest smooths out the overfitting tendencies of single trees while preserving their ability to model complex rules. The MCC of 0.96 confirms it performs reliably across both approved and rejected classes, making it one of the most well-rounded models for this task. |
| **XGBoost (Ensemble)** | XGBoost leads slightly with 98.13% accuracy, edging past Random Forest. It also has the highest AUC (0.999), indicating the best probability calibration among all models. Its precision (0.99) and recall (0.96) are nearly identical to Random Forest. The key difference in approach is that XGBoost builds trees sequentially — each new tree focuses specifically on the applications that previous trees got wrong, which is why it achieves top-tier AUC. The built-in L1/L2 regularization prevents it from overfitting despite its aggressive error-correction strategy. In practice, XGBoost and Random Forest perform almost identically on this dataset, with XGBoost having a marginal edge in both accuracy and probability calibration. |

### Summary
- **XGBoost and Random Forest** are the top performers (~98% accuracy), showing that ensemble methods consistently outperform standalone algorithms on structured financial data.
- **Decision Tree** is surprisingly strong at 97.7%, reflecting the inherently rule-based nature of loan approval decisions. However, ensembles provide better generalization.
- **Logistic Regression** offers a solid 92% baseline, proving that even a simple linear model can go a long way when features are well-structured.
- **KNN** performs reasonably at 89% but is held back by the curse of dimensionality across 12 features.
- **Naive Bayes** struggles significantly (77%) due to its independence assumption — its high precision but extremely low recall means it rejects too many eligible applicants. It would need threshold tuning to be practically useful.
- CIBIL score is likely the single most important feature driving predictions across all models, consistent with real-world banking practices.

---

## Streamlit App Features

1. **CSV Upload** — Upload test data for evaluation
2. **Model Selection** — Dropdown to pick any of the 6 trained classifiers
3. **Metrics Dashboard** — Displays Accuracy, AUC, Precision, Recall, F1, and MCC
4. **Confusion Matrix** — Heatmap showing prediction vs actual results
5. **Classification Report** — Per-class precision, recall, and F1 breakdown
6. **Model Comparison Table** — All 6 models side-by-side with best values highlighted
7. **Predictions Preview** — Individual predictions with probabilities
8. **Feature Importance** — Which features matter most (for tree-based models)

---

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pradeep-kodavoor/Loan-Approval-Prediction.git
   cd Loan-Approval-Prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Get the dataset:**
   - Download from [Kaggle](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
   - Place `loan_approval_dataset.csv` in the project root

4. **Train the models:**
   ```bash
   python model/model_training.py
   ```

5. **Launch the app:**
   ```bash
   streamlit run app.py
   ```

6. **Test:** Upload `test_data.csv` and try different models.

---

## Project Structure

```
Loan-Approval-Prediction/
├── app.py                         # Streamlit web application
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── loan_approval_dataset.csv      # Dataset (download from Kaggle)
├── test_data.csv                  # Evaluation data (auto-generated)
└── model/
    ├── model_training.py          # Training pipeline
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest_ensemble.pkl
    ├── xgboost_ensemble.pkl
    ├── std_scaler.pkl
    ├── feature_columns.pkl
    ├── label_encoders.pkl
    ├── performance_summary.csv
    ├── eda_correlation_heatmap.png
    ├── eda_distributions.png
    ├── eda_boxplots.png
    ├── all_confusion_matrices.png
    ├── metric_comparison_chart.png
    └── feature_importance.png
```

---

## Links

- **GitHub Repository:** [Loan-Approval-Prediction](https://github.com/pradeep-kodavoor/Loan-Approval-Prediction)
- **Live Streamlit App:** [Loan-Approval-Prediction-Service](https://loan-approval-prediction-service.streamlit.app)


---

## References

1. Loan Approval Prediction Dataset — https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset
2. Scikit-learn — https://scikit-learn.org/stable/
3. XGBoost — https://xgboost.readthedocs.io/
4. Streamlit — https://docs.streamlit.io/
