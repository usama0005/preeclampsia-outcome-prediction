"""Shared helpers: data loading, outcome, multiple imputation + pooling, model metrics."""
import os, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, chi2
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score
import config as cfg

warnings.filterwarnings("ignore")


def load_data():
    os.makedirs(f"{cfg.OUT_DIR}/figures", exist_ok=True)
    p = cfg.DATA_PATH
    df = pd.read_csv(p) if p.endswith(".csv") else pd.read_excel(p)
    print(f"Loaded {p}: {df.shape}")
    return df


def save(df, name, index=False):
    df.to_csv(f"{cfg.OUT_DIR}/{name}", index=index)


def code_sets(df):
    """Distinct adverse-outcome codes per patient."""
    return [{int(v) for v in row if v >= 1} for row in df[cfg.OUTCOME_CODE_COLS].fillna(0).to_numpy()]


def make_outcome(df, drop=()):
    """Composite outcome: any code other than the excluded ones."""
    skip = set(cfg.EXCLUDE_CODES) | set(drop)
    return np.array([len(s - skip) > 0 for s in code_sets(df)]).astype(int)


def numeric(df, cols):
    return df[cols].apply(pd.to_numeric, errors="coerce")


# ---------- multiple imputation ----------
def mice_fit(X, y, m):
    """m imputations (outcome included in the imputation model), one logistic fit per imputation."""
    imp, fits, preds = [], [], []
    for i in range(m):
        full = IterativeImputer(sample_posterior=True, random_state=i, max_iter=10).fit_transform(
            np.column_stack([X.to_numpy(), y.astype(float)]))
        Xi = pd.DataFrame(full[:, :X.shape[1]], columns=X.columns)
        r = sm.Logit(y, sm.add_constant(Xi)).fit(disp=0, maxiter=200)
        imp.append(Xi); fits.append(r); preds.append(np.asarray(r.predict(sm.add_constant(Xi))))
    return imp, fits, preds


def rubin(fits):
    """Pool coefficients with Rubin's rules."""
    m = len(fits)
    P = np.array([f.params for f in fits]); S = np.array([f.bse for f in fits])
    beta = P.mean(0); se = np.sqrt((S**2).mean(0) + (1 + 1/m) * P.var(0, ddof=1))
    return beta, se, 2 * norm.sf(np.abs(beta / se))


def coef_table(cols, beta, se, p):
    u = np.array([np.nan] + [cfg.UNITS.get(c, 1) for c in cols]); k = np.nan_to_num(u, nan=1)
    t = pd.DataFrame({"term": ["intercept"] + list(cols), "beta": beta, "SE": se,
                      "OR_per_unit": np.exp(beta * k), "OR_lo": np.exp((beta - 1.96*se) * k),
                      "OR_hi": np.exp((beta + 1.96*se) * k), "unit_increment": u, "p": p})
    t.loc[0, ["OR_per_unit", "OR_lo", "OR_hi", "unit_increment"]] = np.nan
    return t


# ---------- metrics ----------
def nagelkerke(llm, ll0, n):
    c = 1 - np.exp((2/n) * (ll0 - llm))
    return c / (1 - np.exp((2/n) * ll0))


def hl_p(y, pr, g=10):
    d = pd.DataFrame({"y": y, "p": pr}); d["grp"] = pd.qcut(d.p, g, duplicates="drop")
    grp = d.groupby("grp", observed=True)
    o, n, e = grp.y.sum(), grp.y.size(), grp.p.sum()
    return chi2.sf((((o - e)**2) / (e * (1 - e/n))).sum(), len(o) - 2)


def calib(y, pr):
    """Calibration-in-the-large and calibration slope."""
    lp = np.log(pr / (1 - pr))
    citl = sm.Logit(y, np.ones_like(y, float), offset=lp).fit(disp=0).params[0]
    slope = sm.Logit(y, sm.add_constant(lp)).fit(disp=0).params[1]
    return citl, slope


def net_benefit(y, p, ts):
    out = []
    for t in ts:
        d = (p >= t).astype(int); tp = ((d == 1) & (y == 1)).mean(); fp = ((d == 1) & (y == 0)).mean()
        out.append(tp - fp * (t / (1 - t)))
    return np.array(out)


# ---------- full model development ----------
def develop_model(X, y, m=None, B=None, seed=0):
    """MICE + Rubin pooling + apparent metrics + bootstrap optimism correction (model refit every resample).
    Set B=0 to skip the bootstrap."""
    m = m or cfg.M_IMPUTATIONS; B = cfg.N_BOOT if B is None else B
    n = len(y); cols = list(X.columns)
    imp, fits, preds = mice_fit(X, y, m)
    coef = coef_table(cols, *rubin(fits))
    ll0 = sm.Logit(y, np.ones(n)).fit(disp=0).llf
    ap = {k: [] for k in ["AUC", "Brier", "R2N", "HL", "CITL", "slope"]}
    for f, pr in zip(fits, preds):
        ap["AUC"].append(roc_auc_score(y, pr)); ap["Brier"].append(np.mean((pr - y)**2))
        ap["R2N"].append(nagelkerke(f.llf, ll0, n)); ap["HL"].append(hl_p(y, pr))
        c, s = calib(y, pr); ap["CITL"].append(c); ap["slope"].append(s)
    A = {k: np.mean(v) for k, v in ap.items()}

    corr, ci = {}, (np.nan, np.nan)
    if B > 0:
        rng = np.random.default_rng(seed)
        opt = {k: [] for k in ["AUC", "Brier", "slope", "CITL"]}; auc_boot = []
        for Xi in imp:
            D = sm.add_constant(Xi.values)
            for _ in range(B):
                idx = rng.choice(n, n, replace=True)
                if not 5 <= y[idx].sum() <= n - 5: continue
                rb = sm.Logit(y[idx], D[idx]).fit(disp=0, maxiter=200)
                pin, pout = np.asarray(rb.predict(D[idx])), np.asarray(rb.predict(D))
                opt["AUC"].append(roc_auc_score(y[idx], pin) - roc_auc_score(y, pout))
                opt["Brier"].append(np.mean((pin - y[idx])**2) - np.mean((pout - y)**2))
                cb, sb = calib(y[idx], pin); co, so = calib(y, pout)
                opt["slope"].append(sb - so); opt["CITL"].append(cb - co)
                auc_boot.append(roc_auc_score(y[idx], pin))
        ci = np.percentile(auc_boot, [2.5, 97.5])
        corr = {k: A[k] - np.mean(opt[k]) for k in opt}      # corrected = apparent - mean optimism

    r3 = lambda v: round(v, 3) if v is not None and not pd.isna(v) else ""
    metrics = pd.DataFrame([
        ["AUC", r3(A["AUC"]), r3(corr.get("AUC")), f"95% CI {ci[0]:.3f}-{ci[1]:.3f}" if B else ""],
        ["Brier score", r3(A["Brier"]), r3(corr.get("Brier")), "lower is better"],
        ["Nagelkerke R2", r3(A["R2N"]), "", ""],
        ["Calibration-in-the-large", r3(A["CITL"]), r3(corr.get("CITL")), "ideal 0"],
        ["Calibration slope", r3(A["slope"]), r3(corr.get("slope")), "ideal 1"],
        ["Hosmer-Lemeshow p", r3(A["HL"]), "", ">0.05 = adequate fit"]],
        columns=["metric", "apparent", "optimism_corrected", "note"])
    return dict(coef=coef, metrics=metrics, imp=imp, fits=fits, preds=preds,
                pbar=np.mean(preds, axis=0), apparent=A)
