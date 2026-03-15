"""Train multiple classifiers for real-vs-AI detection from feature vectors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classifiers")
    parser.add_argument("--features", type=Path, default=Path("artifacts/feature_vector_classification.csv"))
    parser.add_argument("--metrics-out", type=Path, default=Path("artifacts/results_classification.csv"))
    parser.add_argument("--pred-out", type=Path, default=Path("artifacts/predictions_classification.csv"))
    parser.add_argument("--eval-split", default="val", help="Preferred eval split (val/test)")
    return parser.parse_args()


def split_data(df: pd.DataFrame, eval_split: str):
    if "split" in df.columns and (df["split"] == "train").any():
        train_df = df[df["split"] == "train"].copy()
        if (df["split"] == eval_split).any():
            eval_df = df[df["split"] == eval_split].copy()
        elif (df["split"] == "test").any():
            eval_df = df[df["split"] == "test"].copy()
        else:
            eval_df = df[df["split"] != "train"].copy()
        if len(eval_df) == 0:
            eval_df = train_df.sample(frac=0.2, random_state=42)
            train_df = train_df.drop(eval_df.index)
        return train_df, eval_df

    eval_df = df.sample(frac=0.2, random_state=42)
    train_df = df.drop(eval_df.index)
    return train_df, eval_df


def safe_auc(y_true, y_score) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_auprc(y_true, y_score) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.features)

    meta = {"image_id", "path", "relative_path", "generator", "split", "class_name", "is_real", "y_ai"}
    feat_cols = [c for c in df.columns if c not in meta]

    train_df, eval_df = split_data(df, args.eval_split)

    X_train = train_df[feat_cols].to_numpy(dtype=np.float32)
    y_train = train_df["y_ai"].to_numpy(dtype=np.int64)
    X_eval = eval_df[feat_cols].to_numpy(dtype=np.float32)
    y_eval = eval_df["y_ai"].to_numpy(dtype=np.int64)

    models = {
        "SVM": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
            ]
        ),
        "MLP": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", max_iter=200, random_state=42),
                ),
            ]
        ),
    }

    # XGBoost optional; fallback to sklearn if unavailable.
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss",
                        random_state=42,
                    ),
                ),
            ]
        )
    except Exception:
        pass

    metric_rows = []
    pred_rows = []

    for name, model in models.items():
        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            score = model.predict_proba(X_eval)[:, 1]
        else:
            score = model.decision_function(X_eval)
            score = (score - score.min()) / (score.max() - score.min() + 1e-12)

        pred = (score >= 0.5).astype(np.int64)

        metric_rows.append(
            {
                "model": name,
                "n_train": int(len(train_df)),
                "n_eval": int(len(eval_df)),
                "accuracy": float(accuracy_score(y_eval, pred)),
                "f1": float(f1_score(y_eval, pred, zero_division=0)),
                "auroc": safe_auc(y_eval, score),
                "auprc": safe_auprc(y_eval, score),
            }
        )

        part = eval_df[["image_id", "generator", "split", "class_name", "y_ai"]].copy()
        part["model"] = name
        part["score_ai"] = score
        part["pred_ai"] = pred
        pred_rows.append(part)

    metrics_df = pd.DataFrame(metric_rows).sort_values(by=["auroc", "f1"], ascending=False)
    preds_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.metrics_out, index=False)
    preds_df.to_csv(args.pred_out, index=False)

    print(f"[OK] Models trained: {len(metrics_df)}")
    print(f"[OK] Wrote metrics: {args.metrics_out.resolve()}")
    print(f"[OK] Wrote predictions: {args.pred_out.resolve()}")


if __name__ == "__main__":
    main()
