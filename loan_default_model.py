"""
Loan Default Prediction & Expected Loss Estimation
====================================================
Author  : Oluwadamilola Oyekan, Ph.D.
Contact : oyekan.oa@gmail.com

Description
-----------
Builds and compares multiple classification models to estimate the
Probability of Default (PD) for retail bank loan borrowers.

Expected Loss formula:
    EL = PD × LGD × EAD
    LGD = 1 - Recovery Rate  (Loss Given Default)
    EAD = loan_amount         (Exposure at Default)

Models compared
---------------
1. Logistic Regression      – interpretable baseline
2. Decision Tree            – rule-based, explainable
3. Random Forest            – robust ensemble
4. Gradient Boosting        – high-accuracy ensemble

Usage
-----
    python loan_default_model.py              # train all models, save plots
    python loan_default_model.py --predict    # interactive single-loan predictor
"""

# ---------------------------------------------------------------------------
# 0. Imports
# ---------------------------------------------------------------------------
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Configuration  ← adjust these to match your CSV
# ---------------------------------------------------------------------------
DATA_PATH      = "Task_3_and_4_Loan_Data.csv"
TARGET_COL     = "default"       # 1 = defaulted, 0 = did not default
LOAN_AMT_COL   = "loan_amnt"    # Exposure at Default column name
RECOVERY_RATE  = 0.10           # 10 % recovery rate
RANDOM_STATE   = 42
TEST_SIZE      = 0.20
OUTPUT_DIR     = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 2. Data Loading & Cleaning
# ---------------------------------------------------------------------------

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"\n{'='*60}")
    print(f"  Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"{'='*60}")
    print(f"\nColumn dtypes:\n{df.dtypes.to_string()}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")

    # Drop duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"\nDropped {before - len(df)} duplicate rows.")

    # Target distribution
    print(f"\nTarget distribution:\n{df[TARGET_COL].value_counts().to_string()}")
    print(f"Default rate: {df[TARGET_COL].mean():.2%}")
    return df


# ---------------------------------------------------------------------------
# 3. Feature Engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Debt-to-income
    if "loan_amnt" in df.columns and "income" in df.columns:
        df["debt_to_income"] = df["loan_amnt"] / (df["income"].replace(0, np.nan))

    # Log-transform right-skewed positive columns
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == TARGET_COL:
            continue
        if df[col].skew() > 2 and (df[col] > 0).all():
            df[f"log_{col}"] = np.log1p(df[col])

    return df


def prepare_features(df: pd.DataFrame):
    """Drop target & ID cols; one-hot encode categoricals."""
    y = df[TARGET_COL]
    drop_cols = [TARGET_COL] + [
        c for c in df.columns if c.lower() in {"id", "customer_id", "loan_id", "member_id"}
    ]
    X = df.drop(columns=drop_cols, errors="ignore")

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    print(f"\nFeature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    return X, y


# ---------------------------------------------------------------------------
# 4. Model Pipelines
# ---------------------------------------------------------------------------

def build_pipelines() -> dict:
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("clf",     LogisticRegression(
                max_iter=1000, C=1.0, solver="lbfgs",
                class_weight="balanced", random_state=RANDOM_STATE,
            )),
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf",     DecisionTreeClassifier(
                max_depth=6, min_samples_leaf=20,
                class_weight="balanced", random_state=RANDOM_STATE,
            )),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf",     RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=10,
                class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf",     GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=5,
                subsample=0.8, random_state=RANDOM_STATE,
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# 5. Training & Evaluation
# ---------------------------------------------------------------------------

def evaluate_models(pipelines, X_train, X_test, y_train, y_test) -> dict:
    results = {}
    for name, pipe in pipelines.items():
        print(f"\nFitting {name} ...")
        pipe.fit(X_train, y_train)

        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = pipe.predict(X_test)

        auc   = roc_auc_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        cv_auc = cross_val_score(
            pipe, X_train, y_train,
            cv=StratifiedKFold(5), scoring="roc_auc", n_jobs=-1,
        ).mean()

        results[name] = dict(
            pipeline=pipe, y_prob=y_prob, y_pred=y_pred,
            auc=auc, brier=brier, cv_auc=cv_auc,
        )
        print(f"  AUC={auc:.4f}  CV-AUC={cv_auc:.4f}  Brier={brier:.4f}")
        print(classification_report(y_test, y_pred, digits=4))

    return results


# ---------------------------------------------------------------------------
# 6. Visualisations
# ---------------------------------------------------------------------------

def plot_roc_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set(xlabel="FPR", ylabel="TPR", title="ROC Curves")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_curves.png", dpi=150)
    plt.close()


def plot_confusion_matrices(results, y_test):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        ConfusionMatrixDisplay(cm, display_labels=["No Default", "Default"]).plot(ax=ax)
        ax.set_title(name, fontsize=10)
    plt.suptitle("Confusion Matrices", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=150)
    plt.close()


def plot_feature_importance(results, feature_names):
    for name in ["Random Forest", "Gradient Boosting"]:
        if name not in results:
            continue
        clf = results[name]["pipeline"].named_steps["clf"]
        imp = clf.feature_importances_
        idx = np.argsort(imp)[-20:]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(np.array(feature_names)[idx], imp[idx], color="steelblue")
        ax.set(xlabel="Importance", title=f"Top 20 Features — {name}")
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / f"fi_{name.replace(' ','_').lower()}.png", dpi=150)
        plt.close()


def plot_calibration(results, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in results.items():
        frac, mean_pred = calibration_curve(y_test, res["y_prob"], n_bins=10)
        ax.plot(mean_pred, frac, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set(xlabel="Mean Predicted PD", ylabel="Fraction Positives", title="Calibration Curves")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "calibration.png", dpi=150)
    plt.close()


def plot_el_distribution(df_test, best_probs, loan_amt_col):
    lgd = 1 - RECOVERY_RATE
    ead = df_test[loan_amt_col].values if loan_amt_col in df_test.columns else np.ones(len(best_probs)) * 10_000
    el  = best_probs * lgd * ead

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(el, bins=50, color="tomato", edgecolor="white", alpha=0.85)
    ax.axvline(el.mean(), color="navy", linestyle="--", label=f"Mean EL = ${el.mean():,.0f}")
    ax.set(xlabel="Expected Loss ($)", ylabel="Count", title="Expected Loss Distribution — Test Portfolio")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "el_distribution.png", dpi=150)
    plt.close()

    print(f"\n{'='*50}")
    print("  PORTFOLIO EXPECTED LOSS SUMMARY (Test Set)")
    print(f"{'='*50}")
    print(f"  Loans           : {len(el):,}")
    print(f"  Mean EL         : ${el.mean():,.2f}")
    print(f"  Total EL        : ${el.sum():,.2f}")
    print(f"  Max EL          : ${el.max():,.2f}")
    print(f"  EL as % of EAD  : {el.sum() / ead.sum():.2%}")


# ---------------------------------------------------------------------------
# 7. Expected Loss Function  (primary deliverable)
# ---------------------------------------------------------------------------

def expected_loss(
    loan_features: dict,
    model_pipeline,
    feature_columns: list,
    recovery_rate: float = RECOVERY_RATE,
    loan_amount: float = None,
) -> dict:
    """
    Estimate the Expected Loss for a single loan.

    Parameters
    ----------
    loan_features   : dict mapping feature name → value (same schema as training data)
    model_pipeline  : fitted sklearn Pipeline with predict_proba
    feature_columns : list of feature names in training order
    recovery_rate   : fraction recovered on a defaulted loan (default 10 %)
    loan_amount     : EAD override; if None, read from loan_features[LOAN_AMT_COL]

    Returns
    -------
    dict
        pd  — Probability of Default  (0–1)
        lgd — Loss Given Default      (= 1 − recovery_rate)
        ead — Exposure at Default     ($)
        el  — Expected Loss           (= pd × lgd × ead, $)
    """
    row = pd.DataFrame([loan_features])
    row = pd.get_dummies(row)
    row = row.reindex(columns=feature_columns, fill_value=0)

    pd_val = float(model_pipeline.predict_proba(row)[0, 1])
    lgd    = 1.0 - recovery_rate

    if loan_amount is None:
        loan_amount = float(loan_features.get(LOAN_AMT_COL, 0))

    return {"pd": pd_val, "lgd": lgd, "ead": loan_amount, "el": pd_val * lgd * loan_amount}


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def main():
    df = load_and_clean(DATA_PATH)
    df = engineer_features(df)
    X, y = prepare_features(df)
    feature_cols = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    df_test = df.iloc[X_test.index] if not X_test.index.isin(df.index).all() else df.loc[X_test.index]

    pipelines = build_pipelines()
    results   = evaluate_models(pipelines, X_train, X_test, y_train, y_test)

    # Summary table
    summary = pd.DataFrame([
        {"Model": k, "AUC (test)": v["auc"], "CV-AUC": v["cv_auc"], "Brier": v["brier"]}
        for k, v in results.items()
    ]).set_index("Model")
    print(f"\n{'='*60}\n  MODEL COMPARISON\n{'='*60}")
    print(summary.to_string())
    summary.to_csv(OUTPUT_DIR / "model_comparison.csv")

    best_name = max(results, key=lambda k: results[k]["auc"])
    print(f"\nBest model: {best_name} (AUC={results[best_name]['auc']:.4f})")

    # Plots
    plot_roc_curves(results, y_test)
    plot_confusion_matrices(results, y_test)
    plot_feature_importance(results, feature_cols)
    plot_calibration(results, y_test)
    plot_el_distribution(df_test, results[best_name]["y_prob"], LOAN_AMT_COL)

    # Save best model
    joblib.dump({"pipeline": results[best_name]["pipeline"], "feature_cols": feature_cols},
                OUTPUT_DIR / "best_model.pkl")
    print(f"\nSaved best model → {OUTPUT_DIR}/best_model.pkl")

    # Single-loan demo
    demo = {col: float(X_test.iloc[0][col]) for col in feature_cols}
    demo_ead = float(df_test[LOAN_AMT_COL].iloc[0]) if LOAN_AMT_COL in df_test.columns else 10_000
    el = expected_loss(demo, results[best_name]["pipeline"], feature_cols,
                       loan_amount=demo_ead)
    print(f"\n{'='*50}")
    print("  DEMO — Single Loan Expected Loss")
    print(f"{'='*50}")
    print(f"  PD  : {el['pd']:.4f}  ({el['pd']*100:.2f}%)")
    print(f"  LGD : {el['lgd']:.2f}  ({el['lgd']*100:.0f}%)")
    print(f"  EAD : ${el['ead']:,.2f}")
    print(f"  EL  : ${el['el']:,.2f}")


# ---------------------------------------------------------------------------
# Interactive CLI predictor
# ---------------------------------------------------------------------------

def interactive_predict():
    path = OUTPUT_DIR / "best_model.pkl"
    if not path.exists():
        print("[ERROR] Run without --predict first to train and save a model.")
        return
    artifact     = joblib.load(path)
    pipeline     = artifact["pipeline"]
    feature_cols = artifact["feature_cols"]

    print("\n=== Loan Expected Loss — Interactive Predictor ===")
    loan_features = {}
    for col in feature_cols:
        val = input(f"  {col}: ").strip()
        try:
            loan_features[col] = float(val) if val else 0.0
        except ValueError:
            loan_features[col] = val

    ead = float(input("\n  Loan amount ($): ").strip() or 0)
    rr  = float(input("  Recovery rate (0.10): ").strip() or RECOVERY_RATE)
    res = expected_loss(loan_features, pipeline, feature_cols, rr, ead)

    print(f"\n  PD  : {res['pd']:.4f}")
    print(f"  LGD : {res['lgd']:.4f}")
    print(f"  EAD : ${res['ead']:,.2f}")
    print(f"  EL  : ${res['el']:,.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()
    interactive_predict() if args.predict else main()
