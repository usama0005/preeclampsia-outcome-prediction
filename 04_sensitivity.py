"""Step 4 - sensitivity analyses (missing-data handling, extra predictors, alternative outcomes) and sample-size check."""
import numpy as np, pandas as pd
import statsmodels.api as sm
import config as cfg
from utils import load_data, save, make_outcome, numeric, develop_model

df = load_data(); y = make_outcome(df); CORE = cfg.PREDICTORS

# --- alternative model specifications ---
def summarize(label, cols, imputed=True, note=""):
    X = numeric(df, cols)
    if imputed:
        r = develop_model(X, y, B=0); a = r["apparent"]; n = len(y); coef = r["coef"].set_index("term")
    else:                                                        # complete-case
        from sklearn.metrics import roc_auc_score
        cc = X.notna().all(axis=1).to_numpy(); fit = sm.Logit(y[cc], sm.add_constant(X[cc])).fit(disp=0)
        pp = np.asarray(fit.predict(sm.add_constant(X[cc]))); n = int(cc.sum())
        slope = sm.Logit(y[cc], sm.add_constant(np.log(pp / (1 - pp)))).fit(disp=0).params[1]
        a = dict(AUC=roc_auc_score(y[cc], pp), slope=slope, Brier=np.mean((pp - y[cc])**2)); coef = None
    extra = ""
    if coef is not None and len(cols) > len(CORE):
        extra = "; ".join(f"{c} OR {coef.loc[c, 'OR_per_unit']:.2f} (p={coef.loc[c, 'p']:.2f})" for c in cols[len(CORE):])
    return [label, n, round(a["AUC"], 3), round(a["slope"], 3), round(a["Brier"], 3), extra, note]

rows = [summarize("Primary (MICE)", CORE, note="reference"),
        summarize("Complete-case", CORE, imputed=False),
        summarize(f"Drop {CORE[-1]}", CORE[:-1])]
for name, add in cfg.EXTRA_SETS.items():
    rows.append(summarize(name, CORE + add))
sens = pd.DataFrame(rows, columns=["model", "n", "AUC", "calib_slope", "Brier", "added_predictor_OR(p)", "note"])
save(sens, "sensitivity_models.csv"); print(sens.to_string(index=False))

# --- alternative outcome definitions (refit with the same pipeline, incl. bootstrap) ---
rows = []
for tag, drop in cfg.SENSITIVITY_DROP.items():
    ya = make_outcome(df, drop=drop); r = develop_model(numeric(df, CORE), ya)
    mm = r["metrics"].set_index("metric"); save(r["coef"].round(6), f"coefficients_{tag}.csv")
    rows.append([tag, int(ya.sum()), mm.loc["AUC", "apparent"], mm.loc["AUC", "optimism_corrected"],
                 mm.loc["Calibration slope", "optimism_corrected"]])
alt = pd.DataFrame(rows, columns=["outcome", "events", "AUC_apparent", "AUC_corrected", "slope_corrected"])
save(alt, "sensitivity_outcomes.csv"); print("\n" + alt.to_string(index=False))

# --- minimum sample size for a prediction model (Riley et al. 2020, binary outcome) ---
P, events, N = len(CORE), int(y.sum()), len(y); phi = events / N
R2N = develop_model(numeric(df, CORE), y, B=0)["apparent"]["R2N"]
max_cs = 1 - np.exp(2 * (phi * np.log(phi) + (1 - phi) * np.log(1 - phi)))
R2cs = R2N * max_cs; S = 0.9
n1 = P / ((S - 1) * np.log(1 - R2cs / S))
S2 = R2cs / (R2cs + 0.05 * max_cs); n2 = P / ((S2 - 1) * np.log(1 - R2cs / S2))
n3 = (1.96 / 0.05)**2 * phi * (1 - phi)
nmin = int(np.ceil(max(n1, n2, n3)))
ss = pd.DataFrame({"criterion": ["1: shrinkage <= 0.10", "2: R2 difference <= 0.05", "3: overall-risk margin 0.05",
                                 "minimum N", "your N"], "N": [int(np.ceil(n1)), int(np.ceil(n2)), int(np.ceil(n3)), nmin, N]})
save(ss, "sample_size.csv"); print("\n" + ss.to_string(index=False))
print(f"EPV = {events}/{P} = {events/P:.1f} | N={N} {'meets' if N >= nmin else 'is below'} the minimum of {nmin}")
