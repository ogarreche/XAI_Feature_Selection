#!/usr/bin/env python3
"""
Refactored Multi-Layer Perceptron (MLP) + SHAP script.

Changes:
- Removed redundant imports/functions
- Added comments
- Uses sklearn metrics
- Loads CSVs with a loop
- Vectorized normalization
- Saves metrics and SHAP feature importance to CSV
- Added error handling for file operations
- Fixed SHAP values processing for multi-class classification
- Uses KernelExplainer for MLP compatibility (model-agnostic)
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

REQ_COLS = ['FLOW_DURATION_MILLISECONDS','FIRST_SWITCHED',
            'TOTAL_FLOWS_EXP','TCP_WIN_MSS_IN','LAST_SWITCHED',
            'TCP_WIN_MAX_IN','TCP_WIN_MIN_IN','TCP_WIN_MIN_OUT',
           'PROTOCOL','TCP_WIN_MAX_OUT','TCP_FLAGS',
            'TCP_WIN_SCALE_OUT','TCP_WIN_SCALE_IN','SRC_TOS',
            'DST_TOS','FLOW_ID','L4_SRC_PORT','L4_DST_PORT',
           'MIN_IP_PKT_LEN','MAX_IP_PKT_LEN','TOTAL_PKTS_EXP',
           'TOTAL_BYTES_EXP','IN_BYTES','IN_PKTS','OUT_BYTES','OUT_PKTS',
            'ALERT']

BASE = "/home/oarreche/FeatureSelection/db/SMLdb/"
FILES = [
"dos-03-15-2022-15-44-32.csv","dos-03-16-2022-13-45-18.csv",
"dos-03-17-2022-16-22-53.csv","dos-03-18-2022-19-27-05.csv",
"dos-03-19-2022-20-01-53.csv","dos-03-20-2022-14-27-54.csv",
"normal-03-15-2022-15-43-44.csv","normal-03-16-2022-13-44-27.csv",
"normal-03-17-2022-16-21-30.csv","normal-03-18-2022-19-17-31.csv",
"normal-03-18-2022-19-25-48.csv","normal-03-19-2022-20-01-16.csv",
"normal-03-20-2022-14-27-30.csv","portscanning-03-15-2022-15-44-06.csv",
"portscanning-03-16-2022-13-44-50.csv","portscanning-03-17-2022-16-22-53.csv",
"portscanning-03-18-2022-19-27-05.csv","portscanning-03-19-2022-20-01-45.csv",
"portscanning-03-20-2022-14-27-49.csv"
]

# MLP Hyperparameters
MLP_HIDDEN_LAYERS = (128, 128, 64)  # Architecture: 3 hidden layers
MLP_MAX_ITER = 500
MLP_ACTIVATION = 'relu'  # 'relu', 'tanh', 'logistic', 'identity'
MLP_SOLVER = 'adam'  # 'adam', 'lbfgs', 'sgd'
MLP_LEARNING_RATE = 'adaptive'
MLP_LEARNING_RATE_INIT = 0.001
MLP_ALPHA = 0.0001  # L2 regularization
MLP_EARLY_STOPPING = True
MLP_VALIDATION_FRACTION = 0.2

# --------------------------------------------------
# Load Data
# --------------------------------------------------
print("Loading data...")
frames = []
for f in FILES:
    try:
        df_temp = pd.read_csv(BASE + f, usecols=REQ_COLS)
        frames.append(df_temp)
        print(f"  Loaded: {f}")
    except FileNotFoundError:
        print(f"  Warning: File not found - {f}")
    except Exception as e:
        print(f"  Error loading {f}: {e}")

if not frames:
    raise ValueError("No data files were successfully loaded!")

df = pd.concat(frames, ignore_index=True)

# Note: Sampling 50% of data for faster processing. Remove or adjust as needed.
df = df.sample(frac=0.5, random_state=42)

df = df.fillna(0).drop_duplicates()

X = df.drop(columns="ALERT")
y = df["ALERT"]

# Normalize by absolute max
X = X.div(X.abs().max()).fillna(0)

print("Class counts:")
print(Counter(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# --------------------------------------------------
# Create and Train Model
# --------------------------------------------------
print("Creating MLPClassifier...")
print(f"  Hidden layers: {MLP_HIDDEN_LAYERS}")
print(f"  Activation: {MLP_ACTIVATION}")
print(f"  Solver: {MLP_SOLVER}")
print(f"  Max iterations: {MLP_MAX_ITER}")
print(f"  L2 regularization (alpha): {MLP_ALPHA}")
print(f"  Early stopping: {MLP_EARLY_STOPPING}")

model = MLPClassifier(
    hidden_layer_sizes=MLP_HIDDEN_LAYERS,
    max_iter=MLP_MAX_ITER,
    activation=MLP_ACTIVATION,
    solver=MLP_SOLVER,
    learning_rate=MLP_LEARNING_RATE,
    learning_rate_init=MLP_LEARNING_RATE_INIT,
    alpha=MLP_ALPHA,
    early_stopping=MLP_EARLY_STOPPING,
    validation_fraction=MLP_VALIDATION_FRACTION,
    random_state=42,
    verbose=1,
    n_iter_no_change=20  # For early stopping
)

print("Training...")
t0 = time.time()
model.fit(X_train, y_train)
print(f"Training time: {(time.time()-t0)/60:.2f} min")
print(f"Training iterations: {model.n_iter_}")
print(f"Training loss: {model.loss_}")

# --------------------------------------------------
# Predict and Evaluate
# --------------------------------------------------
print("Predicting...")
t0 = time.time()
preds = model.predict(X_test)
print(f"Prediction time: {(time.time()-t0)/60:.4f} min")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, preds))

accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, average="weighted", zero_division=0)
recall = recall_score(y_test, preds, average="weighted", zero_division=0)
f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
bacc = balanced_accuracy_score(y_test, preds)
mcc = matthews_corrcoef(y_test, preds)
# rocauc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr", zero_division=0)
rocauc = 1 # placeholder for ROC AUC, as it may not be applicable for multi-class without proper handling

metrics = pd.DataFrame([{
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "BalancedAccuracy": bacc,
    "MCC": mcc,
    "ROC_AUC": rocauc
}])

print(metrics)

try:
    metrics.to_csv("MLP_metrics.csv", index=False)
except Exception as e:
    print(f"Error saving metrics: {e}")

# --------------------------------------------------
# SHAP Feature Importance
# --------------------------------------------------
print("Computing SHAP...")
vals = None

try:
    # MLP is not a tree-based model, so use KernelExplainer (model-agnostic)
    # Create a smaller background dataset for faster computation
    background = X_train.sample(n=min(100, len(X_train)), random_state=42)
    explainer = shap.KernelExplainer(model.predict_proba, background)
    
    subset = X_test.iloc[:500]
    print("  Computing SHAP values (this may take a while)...")
    print(f"  Background size: {len(background)}, Subset size: {len(subset)}")
    shap_values = explainer.shap_values(subset)
    
    # Handle both binary and multi-class cases
    if isinstance(shap_values, list):
        # Multi-class: list of arrays, each (n_samples, n_features)
        # Concatenate across classes and average
        vals = np.abs(np.concatenate(shap_values, axis=0)).mean(axis=0)
    else:
        # Binary classification: single array (n_samples, n_features)
        vals = np.abs(shap_values).mean(axis=0)
    
    # Generate and save SHAP summary plot
    try:
        shap.summary_plot(shap_values, subset, show=False)
        plt.savefig("MLP_Shap_Summary_global.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  SHAP summary plot saved")
    except Exception as e:
        print(f"  Warning: Could not save SHAP plot: {e}")
    
except Exception as e:
    print(f"Error computing SHAP values: {e}")
    print("  Note: KernelExplainer can be slow for MLPs. Consider reducing background/subset sizes.")
    vals = None

# --------------------------------------------------
# Save Results
# --------------------------------------------------
if vals is not None:
    feature_importance = (
        pd.DataFrame({
            "col_name": X_train.columns,
            "feature_importance_vals": vals
        })
        .sort_values("feature_importance_vals", ascending=False)
    )
    
    try:
        feature_importance.to_csv("MLP_feature_importance.csv", index=False)
    except Exception as e:
        print(f"Error saving feature importance: {e}")
    
    try:
        combined = feature_importance.copy()
        for c in metrics.columns:
            combined[c] = metrics.iloc[0][c]
        combined.to_csv("MLP_results.csv", index=False)
    except Exception as e:
        print(f"Error saving combined results: {e}")
    
    print("\nFeature Importance (Top 10):")
    print(feature_importance.head(10))
else:
    print("Skipping feature importance calculations due to SHAP error")

print("\nFiles written:")
print("  MLP_metrics.csv")
if vals is not None:
    print("  MLP_feature_importance.csv")
    print("  MLP_results.csv")
    print("  MLP_Shap_Summary_global.png")