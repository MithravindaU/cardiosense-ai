
# ============================================================
# CardioSense-AI: Heart Disease Classification
# Educational ML project, not a medical diagnostic tool
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from google.colab import files

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# 1. Upload and load dataset
uploaded = files.upload()
filename = next(iter(uploaded))
df = pd.read_csv(filename)

print("Dataset shape:", df.shape)
display(df.head())
print("\nMissing values:\n", df.isnull().sum())

# 2. Basic cleaning
df = df.drop(columns=["id", "dataset"], errors="ignore")
df = df.dropna(subset=["num"]).copy()

# Original target:
# 0 = no disease
# 1-4 = disease present
df["target"] = (df["num"] > 0).astype(int)
df = df.drop(columns=["num"])

X = df.drop(columns=["target"])
y = df["target"]

print("\nTarget distribution:")
print(y.value_counts().rename(index={0: "No disease", 1: "Disease present"}))

# 3. Split BEFORE fitting preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Identify numerical and categorical columns
numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

# 5. Preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 6. Random Forest model
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

# 7. Evaluate on held-out test data
y_pred = model.predict(X_test)
y_score = model.predict_proba(X_test)[:, 1]

print("\n===== RANDOM FOREST RESULTS =====")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.3f}")
print(f"F1-score:  {f1_score(y_test, y_pred, zero_division=0):.3f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_score):.3f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["No disease", "Disease present"],
    zero_division=0
))

# 8. Compare with a simple baseline
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

print("\n===== BASELINE COMPARISON =====")
print(f"Baseline accuracy: {accuracy_score(y_test, baseline_pred):.3f}")
print(f"Random Forest accuracy: {accuracy_score(y_test, y_pred):.3f}")

# 9. Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["No disease", "Disease present"],
    cmap="Blues"
)
plt.title("CardioSense-AI: Confusion Matrix")
plt.tight_layout()
plt.show()

# 10. ROC curve
RocCurveDisplay.from_predictions(y_test, y_score)
plt.title("CardioSense-AI: ROC Curve")
plt.tight_layout()
plt.show()

# 11. Feature importance after preprocessing
fitted_preprocessor = model.named_steps["preprocessing"]
classifier = model.named_steps["classifier"]

feature_names = fitted_preprocessor.get_feature_names_out()
importance = pd.Series(
    classifier.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

print("\nTop 15 model features:")
print(importance.head(15))

importance.head(15).sort_values().plot(
    kind="barh",
    figsize=(9, 6)
)
plt.title("Top 15 Random Forest Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# 12. Demonstration prediction using a held-out example
# This is a demonstration, not a real patient's assessment.
example = X_test.iloc[[0]]
predicted_class = model.predict(example)[0]
model_score = model.predict_proba(example)[0, 1]

print("\n===== DEMONSTRATION ONLY =====")
print("Predicted class:", predicted_class)
print("Model-estimated positive-class score:", round(model_score, 3))
print(
    "Class label:",
    "Disease present in dataset label"
    if predicted_class == 1
    else "No disease in dataset label"
)

print(
    "\nDisclaimer: Educational model only. "
    "This output is not a diagnosis or a clinically validated risk estimate."
)


from sklearn.model_selection import StratifiedKFold, cross_validate

# 6.5 Cross-validation on training data only
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring={
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    },
    n_jobs=-1
)

print("\n===== 5-FOLD CROSS-VALIDATION =====")

for metric in [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc"
]:
    scores = cv_results[f"test_{metric}"]
    print(
        f"{metric.replace('_', ' ').title():20s}"
        f"Mean: {scores.mean():.3f} | "
        f"Std: {scores.std():.3f}"
    )


from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# Create a folder for the generated images
assets_dir = Path("assets")
assets_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------------
# 1. CONFUSION MATRIX
# ----------------------------------
fig, ax = plt.subplots(figsize=(6, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels=[0, 1],
    display_labels=["No disease", "Disease present"],
    cmap="Blues",
    values_format="d",
    ax=ax
)

ax.set_title("CardioSense-AI | Confusion Matrix")
fig.tight_layout()
fig.savefig(assets_dir / "confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close(fig)


# ----------------------------------
# 2. ROC CURVE
# ----------------------------------
fig, ax = plt.subplots(figsize=(6, 5))

RocCurveDisplay.from_predictions(
    y_test,
    y_score,
    name="Random Forest",
    ax=ax
)

ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
ax.set_title("CardioSense-AI | ROC Curve")
ax.legend()
fig.tight_layout()
fig.savefig(assets_dir / "roc_curve.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close(fig)


# ----------------------------------
# 3. TOP 15 FEATURE IMPORTANCES
# ----------------------------------
fitted_preprocessor = model.named_steps["preprocessing"]
classifier = model.named_steps["classifier"]

feature_names = fitted_preprocessor.get_feature_names_out()

importance = pd.Series(
    classifier.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

top15 = importance.head(15).sort_values()

fig, ax = plt.subplots(figsize=(9, 6))
top15.plot(kind="barh", ax=ax)

ax.set_title("CardioSense-AI | Top 15 Feature Importances")
ax.set_xlabel("Random Forest importance")
ax.set_ylabel("Encoded input feature")

fig.tight_layout()
fig.savefig(assets_dir / "feature_importance.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close(fig)


# ----------------------------------
# 4. EVALUATION METRICS
# ----------------------------------
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1-score": f1_score(y_test, y_pred, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, y_score)
}

metrics_df = pd.DataFrame(
    {"Metric": list(metrics.keys()), "Score": list(metrics.values())}
)

print(metrics_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(metrics_df["Metric"], metrics_df["Score"])

ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("CardioSense-AI | Test Set Metrics")
ax.bar_label(bars, fmt="%.3f", padding=3)
ax.tick_params(axis="x", rotation=20)

fig.tight_layout()
fig.savefig(assets_dir / "metrics_comparison.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close(fig)


# Save the exact metric values for your README/report
metrics_df.to_csv(assets_dir / "evaluation_metrics.csv", index=False)

print("\nSaved visualizations in:", assets_dir.resolve())
print("Files:", [p.name for p in assets_dir.iterdir()])


import shutil
from google.colab import files

shutil.make_archive("cardiosense_visualizations", "zip", "assets")
files.download("cardiosense_visualizations.zip")
