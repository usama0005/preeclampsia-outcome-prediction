"""Step 2 - primary model: MICE + Rubin pooling, bootstrap optimism correction, nonlinearity, LASSO, incremental value."""
import numpy as np, pandas as pd
import statsmodels.api as sm
from patsy import dmatrix
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
import config as cfg
from utils import load_data, save, make_outcome, numeric, code_sets, develop_model

df = load_data(); y = make_outcome(df); CORE = cfg.PREDICTORS
X0 = numeric(df, CORE)

# --- primary model (multiple imputation, outcome in imputation model) ---
res = develop_model(X0, y)
save(res["coef"].round(6), "model_coefficients.csv"); save(res["metrics"], "model_metrics.csv")
print(res["coef"].round(3).to_string(index=False)); print(); print(res["metrics"].to_string(index=False))
b = res["coef"].beta.to_numpy()
print("\nlogit(P) = %.3f " % b[0] + " ".join(f"{v:+.5f}*{c}" for v, c in zip(b[1:], CORE)))

# --- nonlinearity (LR test: natural cubic spline vs linear) and influence (Cook's D), complete cases ---
cc = X0.notna().all(axis=1); Xc, yc = X0[cc], y[cc.values]
Zc = (Xc - Xc.mean()) / Xc.std(ddof=0); rows = []
for v in CORE:
    others = Zc.drop(columns=[v])
    lin = sm.Logit(yc, sm.add_constant(pd.concat([others, Zc[[v]]], axis=1))).fit(disp=0)
    spl = dmatrix("cr(x, df=3) - 1", {"x": Xc[v].values}, return_type="dataframe"); spl.index = others.index
    nlm = sm.Logit(yc, pd.concat([others, spl], axis=1)).fit(disp=0)
    dfn = spl.shape[1] - 2; lr = 2 * (nlm.llf - lin.llf)
    rows.append([v, round(lr, 2), dfn, round(chi2.sf(lr, dfn), 3)])
nl = pd.DataFrame(rows, columns=["predictor", "LR_stat", "df", "p_nonlinear"]); save(nl, "nonlinearity.csv")
cook = sm.GLM(yc, sm.add_constant(Zc), family=sm.families.Binomial()).fit().get_influence().cooks_distance[0]
print("\nNonlinearity (p>0.05 = linear adequate):\n" + nl.to_string(index=False))
print(f"Max Cook's D = {cook.max():.3f}")

# --- LASSO sensitivity on a wider pool (complete cases, 10-fold CV) ---
Xp = numeric(df, cfg.LASSO_POOL); ok = Xp.notna().all(axis=1)
Zs = StandardScaler().fit_transform(Xp[ok]); yl = y[ok.values]
lasso = LogisticRegressionCV(Cs=30, cv=10, penalty="l1", solver="liblinear", scoring="roc_auc",
                             max_iter=5000, random_state=0).fit(Zs, yl)
sel = [cfg.LASSO_POOL[i] for i in np.where(lasso.coef_[0] != 0)[0]]
pd.Series(sel).to_csv(f"{cfg.OUT_DIR}/lasso_selected.csv", index=False, header=["selected"])
print(f"\nLASSO (n={int(ok.sum())}): selected {len(sel)}/{len(cfg.LASSO_POOL)}: {sel}")
print("Overlap with final model:", sorted(set(sel) & set(CORE)))

# --- incremental value over gestational age (complete cases) ---
def cc_fit(cols, target):
    r = sm.Logit(target, sm.add_constant(Xc[cols])).fit(disp=0, maxiter=200)
    return r, roc_auc_score(target, r.predict(sm.add_constant(Xc[cols])))
base = CORE[0]; r_ga, a_ga = cc_fit([base], yc); r_full, a_full = cc_fit(CORE, yc)
inc = [[f"{base} only", round(a_ga, 3), 0.0]]
for c in CORE[1:]:
    inc.append([f"{base} + {c}", round(cc_fit([base, c], yc)[1], 3), round(cc_fit([base, c], yc)[1] - a_ga, 3)])
inc.append(["full model", round(a_full, 3), round(a_full - a_ga, 3)])
inc = pd.DataFrame(inc, columns=["model", "AUC", "delta_vs_first"]); save(inc, "incremental_value.csv")
lr = 2 * (r_full.llf - r_ga.llf)
print("\n" + inc.to_string(index=False)); print(f"LR test full vs first predictor only: chi2={lr:.1f}, p={chi2.sf(lr, len(CORE)-1):.1e}")

# --- do predictors act in the same direction on maternal vs fetal components? ---
sets = code_sets(df)
for lab, codes in [("maternal", cfg.MATERNAL_CODES), ("fetal", cfg.FETAL_CODES)]:
    t = np.array([bool(s & codes) for s in sets]).astype(int)[cc.values]
    r, a = cc_fit(CORE, t)
    print(f"{lab:9s} AUC {a:.3f} | " + "  ".join(f"{c} {'+' if r.params[c] > 0 else '-'}{'*' if r.pvalues[c] < .05 else ''}" for c in CORE))
