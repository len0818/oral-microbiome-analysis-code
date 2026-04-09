from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


def load_input(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "sample_id"})
    return df


def normalize_target_column(df: pd.DataFrame, requested_target: str) -> str:
    lower_map = {col.lower(): col for col in df.columns}
    if requested_target.lower() not in lower_map:
        raise ValueError(f"Target column '{requested_target}' not found. Available columns: {list(df.columns)}")
    return lower_map[requested_target.lower()]


def sanitize_feature_names(columns) -> list[str]:
    cleaned = []
    for col in columns:
        cleaned.append(str(col).replace("[", "").replace("]", "").replace("<", "").replace(">", ""))
    return cleaned


def run_kruskal_tests(df: pd.DataFrame, target_col: str, output_dir: Path) -> pd.DataFrame:
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [c for c in numeric_columns if c != target_col]

    results = []
    for column in numeric_columns:
        groups = [df[df[target_col] == group][column].dropna() for group in df[target_col].dropna().unique()]
        if len(groups) < 2:
            continue
        stat, pval = kruskal(*groups)
        results.append({"feature": column, "kruskal_statistic": stat, "p_value": pval})

    results_df = pd.DataFrame(results).sort_values("p_value")
    results_df.to_csv(output_dir / "kruskal_wallis_results.csv", index=False)
    return results_df


def build_models() -> Dict[str, GridSearchCV]:
    models = {
        "SVM": GridSearchCV(
            SVC(probability=True),
            param_grid={"C": [0.1, 1, 10], "gamma": [1, 0.1, 0.01], "kernel": ["rbf"]},
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        ),
        "KNN": GridSearchCV(
            KNeighborsClassifier(),
            param_grid={"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan"]},
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        ),
        "DecisionTree": GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            param_grid={
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        ),
        "RandomForest": GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid={
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            },
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        ),
        "XGBoost": GridSearchCV(
            xgb.XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
            ),
            param_grid={
                "n_estimators": [100, 200],
                "max_depth": [3, 6],
                "learning_rate": [0.01, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        ),
    }
    return models


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        ],
        remainder="drop",
    )


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, output_dir: Path) -> None:
    X = X.copy()
    X.columns = sanitize_feature_names(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    preprocessor = make_preprocessor(X_train)
    models = build_models()

    summary_rows = []

    plt.figure(figsize=(8, 6))
    for model_name, grid in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", grid.estimator),
            ]
        )

        # GridSearchCV cannot be nested simply with independent param names unless prefixed
        # Rebuild a prefixed grid for each model
        prefixed_grid = {f"model__{k}": v for k, v in grid.param_grid.items()}
        search = GridSearchCV(
            pipeline,
            prefixed_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        probs = best_model.predict_proba(X_test)[:, 1]
        preds = best_model.predict(X_test)

        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")

        report = classification_report(y_test, preds, output_dict=True)
        summary_rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, preds),
                "auc": roc_auc,
                "best_params": search.best_params_,
                "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
                "precision_class_0": report["0"]["precision"],
                "recall_class_0": report["0"]["recall"],
                "f1_class_0": report["0"]["f1-score"],
                "precision_class_1": report["1"]["precision"],
                "recall_class_1": report["1"]["recall"],
                "f1_class_1": report["1"]["f1-score"],
            }
        )

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    summary_df = pd.DataFrame(summary_rows).sort_values("auc", ascending=False)
    summary_df.to_csv(output_dir / "model_performance_summary.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run genus-level benchmark models.")
    parser.add_argument("--input", required=True, help="Input Excel or CSV file")
    parser.add_argument("--target", default="status", help="Target column name")
    parser.add_argument("--positive", default="severe", help="Positive class label")
    parser.add_argument("--negative", default="mild", help="Negative class label")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_input(args.input)
    target_col = normalize_target_column(df, args.target)

    # Save simple EDA summaries
    numeric_summary = df.select_dtypes(include=[np.number]).describe().T
    numeric_summary.to_csv(output_dir / "numeric_summary.csv")

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    with open(output_dir / "categorical_value_counts.txt", "w", encoding="utf-8") as f:
        for col in categorical_cols:
            f.write(f"[{col}]\n")
            f.write(df[col].value_counts(dropna=False).to_string())
            f.write("\n\n")

    run_kruskal_tests(df, target_col, output_dir)

    filtered = df[df[target_col].astype(str).str.lower().isin([args.positive.lower(), args.negative.lower()])].copy()
    filtered[target_col] = filtered[target_col].astype(str).str.lower()

    X = filtered.drop(columns=[target_col])
    if "sample_id" in X.columns:
        X = X.drop(columns=["sample_id"])
    y = filtered[target_col]

    encoder = LabelEncoder()
    y_encoded = pd.Series(encoder.fit_transform(y), index=y.index)

    train_and_evaluate(X, y_encoded, output_dir)
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
