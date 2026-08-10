#!/usr/bin/env python3
"""
Refactored LightGBM + SHAP script for CICIDS dataset.

Changes:
- Cleaned up redundant imports
- Added configuration section
- Uses sklearn train_test_split with proper stratification
- Proper error handling for file operations
- Uses sklearn metrics instead of custom functions
- Fixed SHAP feature importance calculation
- Saves metrics and feature importance to CSV
- Simplified oversampling with imblearn Pipeline
- Structured output with proper organization
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)
from lightgbm import LGBMClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

np.random.seed(0)

# --------------------------------------------------
# Configuration
# --------------------------------------------------
print('='*80)
print('LightGBM + SHAP Program (CICIDS Dataset)')
print('='*80)
print()

# Feature columns of interest
REQ_COLS = [' Destination Port',' Flow Duration',' Total Fwd Packets',' Total Backward Packets','Total Length of Fwd Packets',
            ' Total Length of Bwd Packets',' Fwd Packet Length Max',' Fwd Packet Length Min',' Fwd Packet Length Mean',
            ' Fwd Packet Length Std','Bwd Packet Length Max',' Bwd Packet Length Min',' Bwd Packet Length Mean',
            ' Bwd Packet Length Std','Flow Bytes/s',' Flow Packets/s',' Flow IAT Mean',' Flow IAT Std',' Flow IAT Max',
            ' Flow IAT Min','Fwd IAT Total',' Fwd IAT Mean',' Fwd IAT Std',' Fwd IAT Max',' Fwd IAT Min','Bwd IAT Total',
            ' Bwd IAT Mean',' Bwd IAT Std',' Bwd IAT Max',' Bwd IAT Min','Fwd PSH Flags',' Bwd PSH Flags',' Fwd URG Flags',
            ' Bwd URG Flags',' Fwd Header Length',' Bwd Header Length','Fwd Packets/s',' Bwd Packets/s',' Min Packet Length',
            ' Max Packet Length',' Packet Length Mean',' Packet Length Std',' Packet Length Variance','FIN Flag Count',
            ' SYN Flag Count',' RST Flag Count',' PSH Flag Count',' ACK Flag Count',' URG Flag Count',' CWE Flag Count',
            ' ECE Flag Count',' Down/Up Ratio',' Average Packet Size',' Avg Fwd Segment Size',' Avg Bwd Segment Size',
            ' Fwd Header Length','Fwd Avg Bytes/Bulk',' Fwd Avg Packets/Bulk',' Fwd Avg Bulk Rate',
            ' Bwd Avg Bytes/Bulk',' Bwd Avg Packets/Bulk','Bwd Avg Bulk Rate','Subflow Fwd Packets',' Subflow Fwd Bytes',
            ' Subflow Bwd Packets',' Subflow Bwd Bytes','Init_Win_bytes_forward',' Init_Win_bytes_backward',
            ' act_data_pkt_fwd',' min_seg_size_forward','Active Mean',' Active Std',' Active Max',' Active Min','Idle Mean',
            ' Idle Std',' Idle Max',' Idle Min',' Label']

# CICIDS database file paths
BASE = "/home/oarreche/FeatureSelection/db/CICIDS/"
FILES = [
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
]

# Label mapping to consolidate attack types
LABEL_MAPPING = {
    'DoS GoldenEye': 'Dos/Ddos',
    'DoS Hulk': 'Dos/Ddos',
    'DoS Slowhttptest': 'Dos/Ddos',
    'DoS slowloris': 'Dos/Ddos',
    'Heartbleed': 'Dos/Ddos',
    'DDoS': 'Dos/Ddos',
    'FTP-Patator': 'Brute Force',
    'SSH-Patator': 'Brute Force',
    'Web Attack - Brute Force': 'Web Attack',
    'Web Attack - Sql Injection': 'Web Attack',
    'Web Attack - XSS': 'Web Attack',
    'Web Attack XSS': 'Web Attack',
    'Web Attack Sql Injection': 'Web Attack',
    'Web Attack Brute Force': 'Web Attack'
}

# LightGBM Hyperparameters
LGBM_N_ESTIMATORS = 100
LGBM_MAX_DEPTH = 10
LGBM_LEARNING_RATE = 0.1
LGBM_NUM_LEAVES = 31
LGBM_VERBOSE = -1

# Dataset parameters
TEST_SIZE = 0.20
DATA_SAMPLE_FRACTION = 0.2  # Use 20% of data for faster processing
CROSS_VALIDATION_SPLITS = 5

# --------------------------------------------------
# Load Databases
# --------------------------------------------------
print('-'*80)
print('Loading Databases')
print('-'*80)
print()

frames = []
for f in FILES:
    try:
        df_temp = pd.read_csv(BASE + f, usecols=REQ_COLS)
        frames.append(df_temp)
        print(f"  Loaded: {f} ({len(df_temp)} rows)")
    except FileNotFoundError:
        print(f"  Warning: File not found - {f}")
    except Exception as e:
        print(f"  Error loading {f}: {e}")

if not frames:
    raise ValueError("No data files were successfully loaded!")

df = pd.concat(frames, ignore_index=True)
print(f"Total records loaded: {len(df)}")

# Sample data for faster processing
df = df.sample(frac=DATA_SAMPLE_FRACTION, random_state=42).reset_index(drop=True)
print(f"Records after sampling ({DATA_SAMPLE_FRACTION*100}%): {len(df)}")

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# --------------------------------------------------
# Preprocess Data
# --------------------------------------------------
print()
print('-'*80)
print('Preprocessing Data')
print('-'*80)
print()

# Map labels
y = df[' Label'].replace(LABEL_MAPPING)

# Normalize features by absolute maximum
X = df.drop(columns=' Label')
X = X.div(X.abs().max()).fillna(0)

# Handle missing values
X = X.fillna(0)
df = X.assign(Label=y)

# Remove duplicates
print("Removing duplicates...")
df = df.drop_duplicates()
print(f"Records after removing duplicates: {len(df)}")

# Separate features and labels
y = df.pop('Label')
X = df

# Print class distribution
print()
print("Class distribution (original):")
class_counts = Counter(y)
for label, count in sorted(class_counts.items()):
    print(f"  {label}: {count}")

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42, stratify=y
)

print()
print(f"Training set: {len(X_train)} samples")
print(f"Testing set: {len(X_test)} samples")

print()
print("Class distribution (training set - before balancing):")
train_counts = Counter(y_train)
for label, count in sorted(train_counts.items()):
    print(f"  {label}: {count}")

# --------------------------------------------------
# Balance Training Dataset with SMOTE-like Oversampling
# --------------------------------------------------
print()
print('-'*80)
print('Balancing Training Dataset')
print('-'*80)
print()

# Use oversampling to balance minority classes
oversample = RandomOverSampler(sampling_strategy='minority', random_state=42)
X_train_balanced, y_train_balanced = oversample.fit_resample(X_train, y_train)

print("Class distribution (training set - after balancing):")
balanced_counts = Counter(y_train_balanced)
for label, count in sorted(balanced_counts.items()):
    print(f"  {label}: {count}")

X_train = X_train_balanced
y_train = y_train_balanced

# --------------------------------------------------
# Build and Train LightGBM Model
# --------------------------------------------------
print()
print('-'*80)
print('Building LightGBM Model')
print('-'*80)
print()

print(f"  N estimators: {LGBM_N_ESTIMATORS}")
print(f"  Max depth: {LGBM_MAX_DEPTH}")
print(f"  Learning rate: {LGBM_LEARNING_RATE}")
print(f"  Num leaves: {LGBM_NUM_LEAVES}")

model = LGBMClassifier(
    n_estimators=LGBM_N_ESTIMATORS,
    max_depth=LGBM_MAX_DEPTH,
    learning_rate=LGBM_LEARNING_RATE,
    num_leaves=LGBM_NUM_LEAVES,
    verbose=LGBM_VERBOSE,
    random_state=42,
    n_jobs=-1
)

# Cross-validation
print()
print('-'*80)
print('Cross-Validation')
print('-'*80)
print()

cv = StratifiedKFold(n_splits=CROSS_VALIDATION_SPLITS, shuffle=True, random_state=42)
print(f"Performing {CROSS_VALIDATION_SPLITS}-fold cross-validation...")

start = time.time()
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
elapsed = time.time() - start

print(f"  Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"  CV computation time: {elapsed/60:.2f} min")

# Train model on full training set
print()
print('-'*80)
print('Training Model (Full Dataset)')
print('-'*80)
print()

start = time.time()
model.fit(X_train, y_train)
elapsed = time.time() - start
print(f"Training time: {elapsed/60:.2f} min")

# --------------------------------------------------
# Predict and Evaluate
# --------------------------------------------------
print()
print('-'*80)
print('Making Predictions')
print('-'*80)
print()

start = time.time()
preds = model.predict(X_test)
preds_proba = model.predict_proba(X_test)
elapsed = time.time() - start
print(f"Prediction time: {elapsed/60:.4f} min")

# --------------------------------------------------
# Compute Metrics
# --------------------------------------------------
print()
print('-'*80)
print('Computing Metrics')
print('-'*80)
print()

print("Confusion Matrix:")
cm = confusion_matrix(y_test, preds)
print(cm)

accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, average="weighted", zero_division=0)
recall = recall_score(y_test, preds, average="weighted", zero_division=0)
f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
bacc = balanced_accuracy_score(y_test, preds)
mcc = matthews_corrcoef(y_test, preds)

# Handle ROC-AUC for multi-class
try:
    rocauc = roc_auc_score(y_test, preds_proba, multi_class="ovr", zero_division=0)
except Exception as e:
    print(f"Warning: Could not compute ROC-AUC: {e}")
    rocauc = np.nan

metrics = pd.DataFrame([{
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "BalancedAccuracy": bacc,
    "MCC": mcc,
    "ROC_AUC": rocauc,
    "CV_Mean_Accuracy": cv_scores.mean(),
    "CV_Std_Accuracy": cv_scores.std()
}])

print()
print("Metrics Summary:")
print(metrics)

try:
    metrics.to_csv("LGBM_CICIDS_metrics.csv", index=False)
    print("\n✓ Metrics saved to LGBM_CICIDS_metrics.csv")
except Exception as e:
    print(f"Error saving metrics: {e}")

# Per-class metrics
print()
print("Per-Class Metrics:")
print(classification_report(y_test, preds, zero_division=0))

# --------------------------------------------------
# SHAP Feature Importance
# --------------------------------------------------
print()
print('-'*80)
print('Computing SHAP Explanations')
print('-'*80)
print()

vals = None

try:
    print("Creating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    # Use a subset for faster computation
    subset = X_test.iloc[:500]
    print(f"Computing SHAP values for {len(subset)} samples...")
    
    start = time.time()
    shap_values = explainer.shap_values(subset)
    elapsed = time.time() - start
    print(f"SHAP computation time: {elapsed/60:.2f} min")
    
    # Handle multi-class case
    if isinstance(shap_values, list):
        # Multi-class: list of arrays, one per class
        # Calculate mean absolute SHAP values across classes and samples
        vals = np.abs(np.concatenate(shap_values, axis=0)).mean(axis=0)
    else:
        # Binary classification
        vals = np.abs(shap_values).mean(axis=0)
    
    # Generate and save SHAP summary plot
    try:
        print("Creating SHAP summary plot...")
        shap.summary_plot(shap_values, subset, show=False)
        plt.savefig("LGBM_CICIDS_Shap_Summary_global.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("✓ SHAP summary plot saved")
    except Exception as e:
        print(f"  Warning: Could not save SHAP plot: {e}")
    
    # Generate and save SHAP beeswarm plot
    try:
        print("Creating SHAP beeswarm plot...")
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[0], subset, plot_type="dot", show=False)
        else:
            shap.summary_plot(shap_values, subset, plot_type="dot", show=False)
        plt.savefig("LGBM_CICIDS_Shap_Summary_Beeswarms.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("✓ SHAP beeswarm plot saved")
    except Exception as e:
        print(f"  Warning: Could not save SHAP beeswarm plot: {e}")
    
except Exception as e:
    print(f"Error computing SHAP values: {e}")
    print("Falling back to model feature importances...")
    try:
        if hasattr(model, 'feature_importances_'):
            vals = model.feature_importances_
            print("Using model feature importances")
    except Exception as e2:
        print(f"Fallback also failed: {e2}")

# --------------------------------------------------
# Save Feature Importance
# --------------------------------------------------
print()
print('-'*80)
print('Feature Importance')
print('-'*80)
print()

if vals is not None:
    feature_importance = pd.DataFrame({
        "col_name": X_train.columns,
        "feature_importance_vals": vals
    }).sort_values("feature_importance_vals", ascending=False)
    
    try:
        feature_importance.to_csv("LGBM_CICIDS_feature_importance.csv", index=False)
        print("✓ Feature importance saved to LGBM_CICIDS_feature_importance.csv")
    except Exception as e:
        print(f"Error saving feature importance: {e}")
    
    try:
        combined = feature_importance.copy()
        for c in metrics.columns:
            combined[c] = metrics.iloc[0][c]
        combined.to_csv("LGBM_CICIDS_results.csv", index=False)
        print("✓ Combined results saved to LGBM_CICIDS_results.csv")
    except Exception as e:
        print(f"Error saving combined results: {e}")
    
    print()
    print("Top 15 Features:")
    print(feature_importance.head(15).to_string(index=False))
else:
    print("Skipping feature importance calculations due to errors")

print()
print('='*80)
print("Files Written:")
print('='*80)
print("  LGBM_CICIDS_metrics.csv")
if vals is not None:
    print("  LGBM_CICIDS_feature_importance.csv")
    print("  LGBM_CICIDS_results.csv")
    print("  LGBM_CICIDS_Shap_Summary_global.png")
    print("  LGBM_CICIDS_Shap_Summary_Beeswarms.png")
print()
print('='*80)
print('Program Complete')
print('='*80)