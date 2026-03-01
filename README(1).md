# Loan Default Prediction & Expected Loss Estimation

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Author:** Oluwadamilola Oyekan, Ph.D. | [LinkedIn](https://linkedin.com/in/oluwadamilola-adeniyi-oyekan-phd-bb6a37130) | oyekan.oa@gmail.com

---

## Project Overview

Retail banks face significant credit risk when borrowers default on personal loans. This project builds a **machine-learning pipeline** that:

1. Predicts the **Probability of Default (PD)** for any borrower given their loan and financial characteristics.
2. Computes the **Expected Loss (EL)** on each loan using the standard credit-risk formula:

$$EL = PD \times LGD \times EAD$$

| Symbol | Meaning | Value |
|--------|---------|-------|
| **PD** | Probability of Default | Predicted by model |
| **LGD** | Loss Given Default = 1 − Recovery Rate | 90% (RR = 10%) |
| **EAD** | Exposure at Default | Loan amount outstanding |

Four classification models are trained and compared:

| Model | Type | Notes |
|-------|------|-------|
| Logistic Regression | Linear | Interpretable baseline, calibrated probabilities |
| Decision Tree | Rule-based | Fully explainable, visualisable |
| Random Forest | Ensemble | Robust, handles non-linearity |
| Gradient Boosting | Ensemble | Highest accuracy, feature importance |

---

## Repository Structure

```
loan-default-prediction/
│
├── Task_3_and_4_Loan_Data.csv   ← input data (not committed to git)
├── loan_default_model.py        ← main training & EL pipeline
├── eda.py                       ← exploratory data analysis
├── requirements.txt             ← Python dependencies
├── outputs/                     ← generated (gitignored)
│   ├── roc_curves.png
│   ├── confusion_matrices.png
│   ├── calibration.png
│   ├── fi_random_forest.png
│   ├── fi_gradient_boosting.png
│   ├── el_distribution.png
│   ├── model_comparison.csv
│   └── best_model.pkl
└── README.md
```

---

## Quickstart

### 1. Clone & install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/loan-default-prediction.git
cd loan-default-prediction
pip install -r requirements.txt
```

### 2. Add your data

Place `Task_3_and_4_Loan_Data.csv` in the project root.

### 3. Run EDA

```bash
python eda.py
```
Charts are saved to `outputs/eda/`.

### 4. Train all models

```bash
python loan_default_model.py
```

This will:
- Preprocess the data and engineer features
- Train & cross-validate four models
- Print a comparison table (AUC, Brier score)
- Save all visualisation plots to `outputs/`
- Save the best model to `outputs/best_model.pkl`
- Print a demo expected-loss calculation

### 5. Interactive single-loan predictor

```bash
python loan_default_model.py --predict
```

You will be prompted to enter borrower characteristics. The script outputs PD, LGD, EAD, and EL for that loan.

### 6. Use the `expected_loss` function in your own code

```python
import joblib
from loan_default_model import expected_loss

artifact     = joblib.load("outputs/best_model.pkl")
pipeline     = artifact["pipeline"]
feature_cols = artifact["feature_cols"]

loan = {
    "income":              55_000,
    "loan_amnt":           15_000,
    "credit_lines_outstanding": 3,
    "loan_to_income":      0.27,
    "years_employed":      4,
    "fico_score":          680,
}

result = expected_loss(
    loan_features   = loan,
    model_pipeline  = pipeline,
    feature_columns = feature_cols,
    recovery_rate   = 0.10,
    loan_amount     = 15_000,
)

print(f"PD  : {result['pd']:.2%}")
print(f"LGD : {result['lgd']:.0%}")
print(f"EAD : ${result['ead']:,.0f}")
print(f"EL  : ${result['el']:,.2f}")
```

---

## Configuration

Open `loan_default_model.py` and edit the **Configuration** block at the top:

```python
DATA_PATH     = "Task_3_and_4_Loan_Data.csv"
TARGET_COL    = "default"       # column name of the binary default indicator
LOAN_AMT_COL  = "loan_amnt"    # column name of loan amount (EAD)
RECOVERY_RATE = 0.10            # 10 % recovery assumption
```

---

## Model Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **AUC-ROC** | Discrimination ability (higher = better; 0.5 = random) |
| **CV-AUC** | 5-fold cross-validated AUC on training set (checks overfitting) |
| **Brier Score** | Calibration quality (lower = better; 0 = perfect) |
| **Confusion Matrix** | Precision/Recall trade-off at 0.5 threshold |
| **Calibration Curve** | Whether predicted probabilities match observed default rates |

---

## Sample Outputs

| Model | AUC (test) | CV-AUC | Brier |
|-------|-----------|--------|-------|
| Logistic Regression | — | — | — |
| Decision Tree | — | — | — |
| Random Forest | — | — | — |
| Gradient Boosting | — | — | — |

*(Run the script to populate with your data.)*

---

## Methodology

### Feature Engineering
- **Debt-to-Income ratio** computed if `loan_amnt` and `income` columns exist.
- **Log-transformations** applied to right-skewed positive features (skewness > 2).
- Categorical variables one-hot encoded.
- Median imputation for missing values.

### Class Imbalance
Default datasets are often imbalanced. All models use `class_weight="balanced"` and results are evaluated on AUC (threshold-independent) rather than accuracy.

### Probability Calibration
The Gradient Boosting model can output poorly calibrated probabilities. For production use, wrap the classifier with `sklearn.calibration.CalibratedClassifierCV` using Platt scaling or isotonic regression.

### Expected Loss
Follows the Basel II/III credit risk framework:

$$\text{EL} = \text{PD} \times \text{LGD} \times \text{EAD}$$

- **PD** is the model's predicted probability.
- **LGD** = 1 − Recovery Rate = 90% (adjustable).
- **EAD** is the outstanding loan balance at time of prediction.

---

## Dependencies

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
joblib>=1.3
```

Install with:
```bash
pip install -r requirements.txt
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

If you use this code in your work, please cite:

```
Oyekan, O. A. (2025). Loan Default Prediction & Expected Loss Estimation.
GitHub. https://github.com/YOUR_USERNAME/loan-default-prediction
```
