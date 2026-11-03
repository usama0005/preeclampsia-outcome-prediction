"""Step 1 - outcome decomposition, candidate predictors, collinearity, missing data."""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import config as cfg
from utils import load_data, save, code_sets, make_outcome, numeric

df = load_data()
y = make_outcome(df); sets = code_sets(df)
print(f"Primary outcome: {y.sum()}/{len(y)} ({y.mean():.1%})")

# --- outcome decomposition ---
comp = pd.DataFrame({"code": list(cfg.CODE_LABELS), "component": list(cfg.CODE_LABELS.values())})
comp["n_patients"] = [sum(c in s for s in sets) for c in comp.code]
mat = np.array([bool(s & cfg.MATERNAL_CODES) for s in sets]); fet = np.array([bool(s & cfg.FETAL_CODES) for s in sets])
summary = pd.Series({"composite (all codes)": sum(len(s) >= 1 for s in sets), "primary composite": int(y.sum()),
                     "any maternal": mat.sum(), "any fetal": fet.sum(), "maternal only": (mat & ~fet).sum(),
                     "fetal only": (fet & ~mat).sum(), "both": (mat & fet).sum()})
save(comp, "outcome_decomposition.csv"); save(summary.to_frame("n"), "outcome_summary.csv", index=True)
print(comp.to_string(index=False)); print(summary.to_string())

# --- candidate predictors: type, missingness, univariable AUC, sparse-binary flag ---
def uni_auc(x, yy):
    m = x.notna(); xv, yo = x[m].to_numpy(), yy[m.to_numpy()]
    n1, n0 = yo.sum(), (yo == 0).sum(); r = stats.rankdata(xv)
    a = (r[yo == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return round(max(a, 1 - a), 3)

rows = []
for c in cfg.CANDIDATES:
    x = pd.to_numeric(df[c], errors="coerce"); nun = x.nunique()
    typ = "binary" if nun <= 2 else "ordinal" if nun <= 5 else "continuous"
    npos = int((x == 1).sum()) if typ == "binary" else None
    ev = int(((x == 1) & (y == 1)).sum()) if typ == "binary" else None
    sparse = typ == "binary" and (npos < 15 or ev < 5)
    rows.append([c, typ, int(x.notna().sum()), round(100 * x.isna().mean(), 1), npos, ev, uni_auc(x, y),
                 "sparse" if sparse else "ok"])
cand = pd.DataFrame(rows, columns=["variable", "type", "n", "pct_missing", "n_pos", "events_in_exposed",
                                   "univ_AUC", "flag"]).sort_values("univ_AUC", ascending=False)
save(cand, "candidate_predictors.csv")
print(f"\nEPV if the whole pool were screened: {y.sum()/len(cfg.CANDIDATES):.1f} | "
      f"EPV for the final model: {y.sum()/len(cfg.PREDICTORS):.1f}")

# --- collinearity (|Spearman r| >= 0.6) among continuous candidates ---
cont = [c for c in cfg.CANDIDATES if pd.to_numeric(df[c], errors="coerce").nunique() > 5]
cm = numeric(df, cont).corr("spearman")
pairs = pd.DataFrame([(a, b, round(cm.loc[a, b], 2)) for i, a in enumerate(cont) for b in cont[i+1:]
                      if abs(cm.loc[a, b]) >= 0.6], columns=["var1", "var2", "spearman_r"])
save(pairs.sort_values("spearman_r", key=abs, ascending=False), "collinearity.csv")
print("\nCollinear pairs:\n" + pairs.to_string(index=False))

# --- missing data: complete-case retention and what missingness is associated with ---
X = numeric(df, cfg.CANDIDATES)
miss = (X.isna().mean() * 100).round(1)
print("\nPredictors with missing values (%):\n" + miss[miss > 0].sort_values(ascending=False).to_string())
for name, cols in {"final model": cfg.PREDICTORS, "all candidates": cfg.CANDIDATES}.items():
    keep = X[cols].notna().all(axis=1)
    print(f"Complete cases, {name}: {keep.sum()}/{len(df)} ({keep.mean():.0%}), events {int(y[keep.to_numpy()].sum())}")

aux = numeric(df, [c for c in cfg.PREDICTORS if X[c].isna().mean() < 0.05]); aux = aux.fillna(aux.median())
rows = []
for c in miss[miss >= 5].index:
    mi = X[c].isna().astype(int)
    p_out = stats.fisher_exact(pd.crosstab(mi, y).values)[1]
    auc = roc_auc_score(mi, LogisticRegression(max_iter=1000).fit(aux, mi).predict_proba(aux)[:, 1])
    rows.append([c, miss[c], round(auc, 2), round(p_out, 3)])
mt = pd.DataFrame(rows, columns=["variable", "pct_missing", "AUC_miss_vs_observed", "p_miss_vs_outcome"])
save(mt, "missingness.csv"); print("\nMissingness associations:\n" + mt.to_string(index=False))
print("(Association with observed data is consistent with MAR; association with the outcome flags possibly informative missingness.)")
