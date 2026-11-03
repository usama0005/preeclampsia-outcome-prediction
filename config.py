"""Settings for the pipeline. Edit this file to point it at your own data."""
import os

DATA_PATH = os.environ.get("DATA_PATH", "data/your_data.xlsx")   # .xlsx or .csv
OUT_DIR = "outputs"

# --- outcome: composite of adverse-event codes stored across several columns ---
OUTCOME_CODE_COLS = [f"adverse_outcome_{i}" for i in range(1, 6)]
EXCLUDE_CODES = {17}                     # codes not counted in the primary outcome (e.g. delivery <32 wk)
MATERNAL_CODES = set(range(1, 13))
FETAL_CODES = {13, 14, 15, 16}
SENSITIVITY_DROP = {"noFGR": {13}, "noFGR_noMalformation": {13, 16}}   # alternative composites
CODE_LABELS = {1: "Cerebrovascular accident", 2: "Retinal detachment", 3: "Heart failure",
               4: "Pulmonary edema", 5: "Liver injury", 6: "Acute kidney injury", 7: "Eclampsia",
               8: "Placental abruption", 9: "HELLP", 10: "Postpartum hemorrhage", 11: "DIC",
               12: "Maternal death", 13: "Fetal growth restriction", 14: "Intrauterine distress",
               15: "Stillbirth", 16: "Malformation", 17: "Delivery <32 weeks"}

# --- predictors ---
CANDIDATES = ["age", "gest_age", "bmi_adm", "sbp", "dbp", "fundal_height_cm", "abdominal_circ_cm",
              "pregnancy_count", "live_births", "abortions", "edema_grade", "prenatal_checkup",
              "menstrual_history", "clin_manifest_binary", "is_twin", "assisted_reproduction",
              "hx_HTN", "hx_gdm", "hx_cesarean", "hx_hep_b", "hx_pcos", "hx_macrosomia", "adv_preg_hx",
              "WBC", "RBC", "Hb", "PLT", "APTT", "FIB", "d_dimer", "Alb", "TBIL", "ALT", "AST", "LDH",
              "fasting_BGC", "BUN", "Cr", "UA", "HSCRP", "UP"]
PREDICTORS = ["gest_age", "LDH", "dbp", "bmi_adm", "fasting_BGC"]          # final model
UNITS = {"gest_age": 1, "LDH": 10, "dbp": 5, "bmi_adm": 1, "fasting_BGC": 1}  # OR reported per increment
LABELS = {"gest_age": "GA at admission, per week", "LDH": "LDH, per 10 U/L", "dbp": "Diastolic BP, per 5 mmHg",
          "bmi_adm": "Admission BMI, per kg/m²", "fasting_BGC": "Fasting glucose, per mmol/L"}
LASSO_POOL = ["gest_age", "dbp", "bmi_adm", "fundal_height_cm", "LDH", "AST", "Alb", "TBIL", "BUN", "Cr", "UA",
              "WBC", "Hb", "PLT", "APTT", "FIB", "d_dimer", "fasting_BGC", "age", "pregnancy_count",
              "edema_grade", "hx_gdm", "hx_cesarean", "is_twin", "clin_manifest_binary", "prenatal_checkup"]
EXTRA_SETS = {"+ UP": ["UP"], "+ TC + TG": ["TC", "TG"]}                  # sensitivity: add predictors

# --- validation ---
YEAR_COL = "adm_year"                    # set to None to skip temporal validation
TEMPORAL_CUT = 2019                      # develop <= cut, validate > cut
THRESHOLDS = [0.10, 0.20, 0.30, 0.40]

# --- runtime (set SMOKE=1 for a fast test run) ---
SMOKE = bool(os.environ.get("SMOKE"))
M_IMPUTATIONS = 3 if SMOKE else 20
N_BOOT = 10 if SMOKE else 100            # bootstrap resamples per imputation
N_BOOT_CI = 100 if SMOKE else 1000       # bootstrap resamples for threshold CIs
