import json
import os

with open('kaggle_notebook_final.py', 'r') as f:
    py_code = f.read()

# Sections definition for Final Notebook
raw_sections = [
    ("# 1. IMPORTS & SETUP", "## 1. Imports and Environment Setup", "Import core machine learning, preprocessing, and gradient boosting frameworks (CatBoost, LightGBM, XGBoost, and scikit-learn)."),
    ("# 2. LOAD COMPETITION DATA", "## 2. Load Competition Data", "Dynamically locate and ingest raw competition CSVs across Kaggle `/kaggle/input` directories or local workspace."),
    ("# 3. FEATURE ENGINEERING", "## 3. Feature Engineering (86 Base Features + 4 Encodings = 90 Features)", "Extract all 86 base features (multi-window activity aggregations, recency metrics, weekly profiles) plus 4 leakage-safe target/frequency encodings."),
    ("# 4. NESTED INTERNAL OOF STACKING", "## 4. Nested 5-Fold Internal OOF Generation on 60,000 Training Rows", "Generate strictly leakage-safe out-of-fold probability predictions across all 5 base models to prepare the Level-2 meta-feature matrix."),
    ("# 5. FIT V17 LEVEL-2 META-LEARNER", "## 5. Fit V17 Level-2 Meta-Learner & Estimate Residual Bias", "Train the regularized Logistic Regression meta-model (C=0.1) on the genuine OOF probability matrix, and compute training residual biases for segment calibration."),
    ("# 6. RETRAIN BASE MODELS", "## 6. Retrain Base Models on Full 60,000 Training Rows", "Fit the 5-model zoo (CatBoost, LightGBM, XGBoost, Specialist, Logistic Regression) on all 60,000 rows and generate 40,000 test predictions."),
    ("# 7. ASSEMBLE FINAL COMPOSITE PREDICTIONS", "## 7. Assemble Final Winning Composite (80% V15 Champion + 20% V17 Stack)", "Combine the V15 segment-calibrated tree candidate with the V17 learned meta-model predictions."),
    ("# 8. VALIDATE SUBMISSION FORMAT", "## 8. Validate Submission Format & Integrity Checks", "Run 6 comprehensive integrity checks ensuring exactly 40,000 rows, proper headers, valid probability bounds, and zero NaNs."),
    ("# 9. SAVE SUBMISSION.CSV", "## 9. Save submission.csv", "Export verified predictions to `submission.csv` for Kaggle leaderboard evaluation.")
]

cells = []

# Title markdown
intro_md = (
    "# Inspirathon - Customer Churn Prediction: Final Production Submission Pipeline\n\n"
    "This self-contained Kaggle notebook reproduces the **Final Champion Pipeline (80% V15 Champion + 20% V17 Stacked Model)** directly from raw competition CSVs.\n\n"
    "### Pipeline Architecture\n"
    "- **Pure Competition Ingestion**: Ingests `train.csv`, `test.csv`, `events.csv`, and `sample_submission.csv`.\n"
    "- **Zero External File Dependencies**: All features and weights computed end-to-end within the notebook.\n"
    "- **Strict Temporal Windowing**: All events filtered on `timestamp <= snapshot_date`.\n"
    "- **90 Curated Features**:\n"
    "  * 86 Base Features: Multi-window rollups (7, 14, 30, 60, 90, 120d), recencies ($k^{\\text{th}}$-last sessions), activity ratios, and 8-week activity profiles/slopes.\n"
    "  * 4 Leakage-Safe Target Encodings: Out-of-fold `referral_code` TE ($m=20.0$), training frequency, and activity/recency coarse group encodings.\n"
    "- **Level-1 Base Model Zoo (Retrained on 60k rows)**:\n"
    "  * CatBoost Depth-3 (5 Seeds)\n"
    "  * LightGBM Depth-3 (5 Seeds)\n"
    "  * XGBoost Depth-3 (5 Seeds)\n"
    "  * Billing Specialist (5 Seeds on Monthly + Annual)\n"
    "  * Regularized Logistic Regression ($C=0.1$)\n"
    "- **Level-2 Meta-Learner**: Regularized Logistic Regression ($C=0.1$) trained strictly on 5-fold internal OOF base predictions.\n"
    "- **V15 Segment Calibration**: Post-hoc residual calibration for Central Region ($-0.0117$) and Annual Billing ($-0.0048$).\n"
    "- **Final Winning Composite Blend**: 80% V15 Champion + 20% V17 Stack.\n"
    "- **Verified Temporal CV AUC**: **0.86939** (Highest verified across entire campaign; Fold 1: 0.87282, Fold 2: 0.86340, Fold 3: 0.87195).\n"
    "- **Runtime**: ~3-4 minutes end-to-end on CPU."
)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [intro_md]
})

indices = []
for marker, title, desc in raw_sections:
    pos = py_code.find(marker)
    assert pos != -1, f"Marker {marker} not found in kaggle_notebook_final.py!"
    indices.append(pos)

for i in range(len(raw_sections)):
    marker, title, desc = raw_sections[i]
    start_pos = indices[i]
    end_pos = indices[i + 1] if i + 1 < len(raw_sections) else len(py_code)
    section_code = py_code[start_pos:end_pos].strip()
    
    code_lines = section_code.splitlines(keepends=True)
    filtered_code = []
    past_header = False
    for line in code_lines:
        if not past_header:
            if line.startswith("#") and any(k in line for k in ["====", marker, "IMPORTS", "LOAD", "FEATURE", "NESTED", "TRAINING", "RETRAIN", "ASSEMBLE", "VALIDATE", "SAVE"]):
                continue
            else:
                past_header = True
        filtered_code.append(line)
    
    clean_code = "".join(filtered_code).strip()
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"{title}\n\n{desc}"]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [clean_code]
    })

nb = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.10"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("kaggle_notebook_final.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Generated kaggle_notebook_final.ipynb with clean structure.")
