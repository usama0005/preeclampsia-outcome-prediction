# Preeclampsia Adverse-Outcome Prediction

A short, generic Python pipeline for developing and validating a clinical prediction model for adverse
maternal/fetal outcomes in preeclampsia. It uses multiple imputation (MICE) with logistic regression,
bootstrap internal validation, temporal validation, decision-curve analysis, sensitivity analyses and a nomogram.

**No data is included.** Bring your own dataset and edit `config.py` to match your column names.
Results will depend on your data and will not be identical to any published numbers.

## Pipeline

| Script | What it does |
|---|---|
| `01_descriptives.py` | Composite-outcome breakdown, candidate predictor table (univariable AUC), collinearity, missing-data summary |
| `02_model.py` | Final model with MICE (m=20) + Rubin's rules, bootstrap optimism-corrected AUC/Brier/calibration, nonlinearity test, LASSO check, incremental value |
| `03_validation.py` | Sensitivity/specificity/PPV/NPV at fixed thresholds, decision curves, temporal validation, ROC / calibration / DCA / forest plots |
| `04_sensitivity.py` | Complete-case vs MICE, extra predictors, alternative outcome definitions, minimum sample size (Riley et al.) |
| `05_nomogram.py` | Points-based nomogram from the pooled coefficients |

Shared code lives in `utils.py`; all settings (data path, columns, predictors, outcome codes) are in `config.py`.

## Setup

```bash
pip install -r requirements.txt
```

## Your data

A single `.xlsx` or `.csv` file with one row per patient, placed in `data/` (git-ignored). It should contain:

- adverse-outcome code columns (`OUTCOME_CODE_COLS`; 0 = none, 1–17 = event codes). The primary outcome is "any code except `EXCLUDE_CODES`"
- the predictor columns listed in `CANDIDATES` / `PREDICTORS`
- optionally an admission-year column (`YEAR_COL`) for temporal validation (set to `None` to skip)

Rename columns or change the predictor list in `config.py` to fit your dataset.

## Run

Quick test first (few imputations/bootstraps, about a minute):

```bash
SMOKE=1 DATA_PATH=data/your_data.xlsx bash run_all.sh
```

Full run:

```bash
DATA_PATH=data/your_data.xlsx bash run_all.sh
```

Or run any step alone, e.g. `python 02_model.py`. Tables go to `outputs/`, figures to `outputs/figures/`.

## Notes

- The final model is a parsimonious model developed on the data, so interpret the selection and its estimates with that in mind.
- Patient data and results derived from it are excluded from git via `.gitignore`. Do not commit them.
- Python 3.9+.
