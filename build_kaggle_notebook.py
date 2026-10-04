import json
import os

with open('kaggle_notebook_v5.py', 'r') as f:
    py_code = f.read()

# Split kaggle_notebook_v5.py into the 7 sections cleanly
# Section markers:
# # ===========================================================================
# # 1. IMPORTS
# ...
# # 2. LOAD COMPETITION DATA
# ...
# # 3. FEATURE ENGINEERING (77 V5 FEATURES)
# ...
# # 4. TRAIN V5 ENSEMBLE (5-SEED CATBOOST)
# ...
# # 5. PREDICT TEST (ENSEMBLE AVERAGE)
# ...
# # 6. VALIDATE SUBMISSION FORMAT
# ...
# # 7. SAVE SUBMISSION.CSV

sections_def = [
    {
        "title": "# Inspirathon - Customer Churn Prediction: V5 Production Pipeline",
        "desc": (
            "This self-contained Kaggle notebook reproduces the **V5 Winning Pipeline** directly from the competition CSVs.\n\n"
            "### Pipeline Highlights\n"
            "- **Pure Competition Data**: Ingests `train.csv`, `test.csv`, `events.csv`, and `sample_submission.csv`.\n"
            "- **Strict Temporal Windowing**: All events filtered on `timestamp <= snapshot_date`.\n"
            "- **77 Curated Features**: Recency progression ($k^{\\text{th}}$-last sessions), multi-window aggregations (7, 14, 30, 60, 90, 120d), acceleration deltas, and inactivity ratio.\n"
            "- **5-Seed CatBoost Ensemble**: Random seeds `[42, 123, 314, 555, 777]` with calibrated iterations.\n"
            "- **Runtime**: ~60 seconds end-to-end on CPU.\n"
            "- **Temporal CV AUC**: Verified **0.86448** mean AUC across 3 forward temporal folds."
        )
    },
    {
        "marker": "# 1. IMPORTS",
        "title": "## 1. Imports and Environment Setup",
        "desc": "Import core data manipulation and modeling libraries."
    },
    {
        "marker": "# 2. LOAD COMPETITION DATA",
        "title": "## 2. Load Competition Data",
        "desc": "Dynamically locate competition files across Kaggle input directories or local workspace."
    },
    {
        "marker": "# 3. FEATURE ENGINEERING",
        "title": "## 3. Feature Engineering (77 Curated Features)",
        "desc": "Extract event aggregations, recencies, activity densities, window ratios, and profile interactions."
    },
    {
        "marker": "# 4. TRAIN V5 ENSEMBLE",
        "title": "## 4. Train V5 Ensemble (5-Seed CatBoost)",
        "desc": "Train 5-seed CatBoost ensemble on all 60,000 training rows using optimal V5 hyperparameters."
    },
    {
        "marker": "# 5. PREDICT TEST",
        "title": "## 5. Predict Test (Ensemble Average)",
        "desc": "Compute ensemble average probabilities across all 5 seeds."
    },
    {
        "marker": "# 6. VALIDATE SUBMISSION FORMAT",
        "title": "## 6. Validate Submission Format",
        "desc": "Run 6 comprehensive integrity checks on the submission DataFrame."
    },
    {
        "marker": "# 7. SAVE SUBMISSION.CSV",
        "title": "## 7. Save submission.csv",
        "desc": "Export continuous churn predictions to `submission.csv`."
    }
]

# Parse sections precisely based on markers
raw_sections = [
    ("# 1. IMPORTS", "## 1. Imports and Environment Setup", "Import core data manipulation and modeling libraries."),
    ("# 2. LOAD COMPETITION DATA", "## 2. Load Competition Data", "Dynamically locate competition files across Kaggle input directories or local workspace."),
    ("# 3. FEATURE ENGINEERING", "## 3. Feature Engineering (77 Curated Features)", "Extract event aggregations, recencies, activity densities, window ratios, and profile interactions."),
    ("# 4. TRAIN V5 ENSEMBLE", "## 4. Train V5 Ensemble (5-Seed CatBoost)", "Train 5-seed CatBoost ensemble on all 60,000 training rows using optimal V5 hyperparameters."),
    ("# 5. PREDICT TEST", "## 5. Predict Test (Ensemble Average)", "Compute ensemble average probabilities across all 5 seeds."),
    ("# 6. VALIDATE SUBMISSION FORMAT", "## 6. Validate Submission Format", "Run 6 comprehensive integrity checks on the submission DataFrame."),
    ("# 7. SAVE SUBMISSION.CSV", "## 7. Save submission.csv", "Export continuous churn predictions to `submission.csv`.")
]

# Split py_code by section comments
cells = []

# Title markdown
intro_md = (
    "# Inspirathon - Customer Churn Prediction: V5 Production Pipeline\n\n"
    "This self-contained Kaggle notebook reproduces the **V5 Winning Pipeline** directly from the competition CSVs.\n\n"
    "### Pipeline Highlights\n"
    "- **Pure Competition Data**: Ingests `train.csv`, `test.csv`, `events.csv`, and `sample_submission.csv`.\n"
    "- **Strict Temporal Windowing**: All events filtered on `timestamp <= snapshot_date`.\n"
    "- **77 Curated Features**: Recency progression ($k^{\\text{th}}$-last sessions), multi-window aggregations (7, 14, 30, 60, 90, 120d), acceleration deltas, and inactivity ratio.\n"
    "- **5-Seed CatBoost Ensemble**: Random seeds `[42, 123, 314, 555, 777]` with calibrated iterations.\n"
    "- **Runtime**: ~60 seconds end-to-end on CPU.\n"
    "- **Temporal CV AUC**: Verified **0.86448** mean AUC across 3 forward temporal folds."
)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [intro_md]
})

indices = []
for marker, title, desc in raw_sections:
    pos = py_code.find(marker)
    assert pos != -1, f"Marker {marker} not found!"
    indices.append(pos)

for i in range(len(raw_sections)):
    marker, title, desc = raw_sections[i]
    start_pos = indices[i]
    end_pos = indices[i + 1] if i + 1 < len(raw_sections) else len(py_code)
    section_code = py_code[start_pos:end_pos].strip()
    
    # Remove leading comments line if desired, or keep as is
    code_lines = section_code.splitlines(keepends=True)
    # Filter out initial decorative lines like # ==== and the marker itself
    filtered_code = []
    past_header = False
    for line in code_lines:
        if not past_header:
            if line.startswith("#") and any(k in line for k in ["====", marker, "IMPORTS", "LOAD", "FEATURE", "TRAIN", "PREDICT", "VALIDATE", "SAVE"]):
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

with open("kaggle_notebook_v5.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Generated kaggle_notebook_v5.ipynb with clean extraction.")
