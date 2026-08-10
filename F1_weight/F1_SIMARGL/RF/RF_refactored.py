\
#!/usr/bin/env python3
"""
Refactored Random Forest + SHAP script.

Changes:
- Removed redundant imports/functions
- Added comments
- Uses sklearn metrics
- Loads CSVs with a loop
- Vectorized normalization
- Saves metrics and SHAP feature importance to CSV
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
# REQ_COLS = [
#     'FLOW_DURATION_MILLISECONDS','FIRST_SWITCHED','TOTAL_FLOWS_EXP',
#     'TCP_WIN_MSS_IN','LAST_SWITCHED','TCP_WIN_MAX_IN','TCP_WIN_MIN_IN',
#     'TCP_WIN_MIN_OUT','PROTOCOL','TCP_WIN_MAX_OUT','TCP_FLAGS',
#     'TCP_WIN_SCALE_OUT','TCP_WIN_SCALE_IN','SRC_TOS','DST_TOS','ALERT'
# ]

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

print("Loading data...")
frames = [pd.read_csv(BASE + f, usecols=REQ_COLS) for f in FILES]
df = pd.concat(frames, ignore_index=True).sample(frac=0.5, random_state=42)

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

rf = RandomForestClassifier(
    max_depth=10,
    n_estimators=50,
    min_samples_split=2,
    n_jobs=-1,
    random_state=42
)

print("Training...")
t0 = time.time()
rf.fit(X_train, y_train)
print(f"Training time: {(time.time()-t0)/60:.2f} min")

print("Predicting...")
t0 = time.time()
preds = rf.predict(X_test)
print(f"Prediction time: {(time.time()-t0)/60:.4f} min")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, preds))

accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, average="weighted")
recall = recall_score(y_test, preds, average="weighted")
f1 = f1_score(y_test, preds, average="weighted")
bacc = balanced_accuracy_score(y_test, preds)
mcc = matthews_corrcoef(y_test, preds)
rocauc = roc_auc_score(y_test, rf.predict_proba(X_test), multi_class="ovr")

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

metrics.to_csv("RF_metrics.csv", index=False)

print("Computing SHAP...")
explainer = shap.TreeExplainer(rf)
subset = X_test.iloc[:500]
shap_values = explainer.shap_values(subset)

shap.summary_plot(shap_values, subset, show=False)
plt.savefig("RF_Shap_Summary_global.png", dpi=300, bbox_inches="tight")
plt.close()

vals = np.abs(np.asarray(shap_values)).mean(axis=(0,1))

feature_importance = (
    pd.DataFrame({
        "col_name": X_train.columns,
        "feature_importance_vals": vals
    })
    .sort_values("feature_importance_vals", ascending=False)
)

feature_importance.to_csv("RF_feature_importance.csv", index=False)

combined = feature_importance.copy()
for c in metrics.columns:
    combined[c] = metrics.iloc[0][c]
combined.to_csv("RF_results.csv", index=False)

print(feature_importance)
print("\nFiles written:")
print("  RF_metrics.csv")
print("  RF_feature_importance.csv")
print("  RF_results.csv")
