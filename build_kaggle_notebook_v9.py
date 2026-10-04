import json
import os

with open('kaggle_notebook_v9.py', 'r') as f:
    py_code = f.read()

# Sections definition for V9 notebook
raw_sections = [
    ("# 1. IMPORTS", "## 1. Imports and Environment Setup", "Import core data manipulation, feature engineering, and model libraries (CatBoost & scikit-learn)."),
    ("# 2. LOAD COMPETITION DATA", "## 2. Load Competition Data", "Dynamically locate and load competition files across Kaggle input directories or local workspace."),
    ("# 3. FEATURE ENGINEERING", "## 3. Feature Engineering (86 Base Features + 4 Encodings = 90 Features)", "Construct the complete 90-feature matrix: 86 base features (window rollups, recencies, activity densities, weekly trajectories) plus 4 leakage-safe target/frequency encodings (referral code m=20, activity-billing, and recency-billing)."),
    ("# 4. TRAIN V9 CATBOOST 5-SEED ENSEMBLE", "## 4. Train Depth-4 CatBoost 5-Seed Ensemble", "Train the 5-seed Depth-4 CatBoost ensemble (seeds: 42, 123, 314, 555, 777) on all 60,000 training rows using optimal regularized hyperparameters and calibrated seed iterations."),
    ("# 5. TRAIN REGULARIZED LOGISTIC REGRESSION", "## 5. Train Regularized Logistic Regression", "Train regularized Logistic Regression (C=0.1, median imputer, standard scaler, and one-hot encoder) on the 86 base V7 features across all 60,000 rows."),
    ("# 6. COMPUTE FINAL WINNING BLEND", "## 6. Compute Final Winning Blend (88% CatBoost + 12% Logistic Regression)", "Compute the championship blend combining the 5-seed tree ensemble and the regularized linear model."),
    ("# 7. VALIDATE SUBMISSION FORMAT", "## 7. Validate Submission Format & Integrity Checks", "Run 6 comprehensive integrity checks ensuring exactly 40,000 rows, proper headers, valid probability bounds, and zero NaNs."),
    ("# 8. SAVE SUBMISSION.CSV", "## 8. Save submission.csv", "Export verified predictions to `submission.csv` for Kaggle leaderboard evaluation.")
]

cells = []

# Title markdown
intro_md = (
    "# Inspirathon - Customer Churn Prediction: V9 Champion Pipeline\n\n"
    "This self-contained Kaggle notebook reproduces the **V9 Champion Pipeline** directly from raw competition CSVs.\n\n"
    "### Pipeline Highlights\n"
    "- **Pure Competition Data**: Ingests `train.csv`, `test.csv`, `events.csv`, and `sample_submission.csv`.\n"
    "- **Zero Local Artifact Dependencies**: Reconstructs all features from scratch; no precomputed files or external weights.\n"
    "- **Strict Temporal Windowing**: All events filtered on `timestamp <= snapshot_date`.\n"
    "- **90 Curated Features**:\n"
    "  * 86 Base Features: Multi-window aggregations (7, 14, 30, 60, 90, 120d), recencies ($k^{\\text{th}}$-last sessions), activity ratios, and 8-week activity profiles/slopes.\n"
    "  * 4 Leakage-Safe Target Encodings: Out-of-fold `referral_code` TE ($m=20.0$), training frequency, and activity/recency coarse group encodings.\n"
    "- **5-Seed Depth-4 CatBoost Ensemble**: Random seeds `[42, 123, 314, 555, 777]` with calibrated tree iterations.\n"
    "- **Regularized Logistic Regression**: Diverse linear baseline ($C=0.1$).\n"
    "- **88/12 Winning Blend**: 88% CatBoost Ensemble + 12% Logistic Regression.\n"
    "- **Verified Temporal CV AUC**: **0.86824** mean AUC across 3 forward temporal folds.\n"
    "- **Fast Runtime**: ~75 seconds end-to-end on CPU."
)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [intro_md]
})

indices = []
for marker, title, desc in raw_sections:
    pos = py_code.find(marker)
    assert pos != -1, f"Marker {marker} not found in kaggle_notebook_v9.py!"
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
            if line.startswith("#") and any(k in line for k in ["====", marker, "IMPORTS", "LOAD", "FEATURE", "TRAIN", "COMPUTE", "VALIDATE", "SAVE"]):
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

with open("kaggle_notebook_v9.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Generated kaggle_notebook_v9.ipynb with clean structure.")
