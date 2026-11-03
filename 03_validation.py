"""Step 3 - thresholds, decision curve, temporal validation, ROC / calibration / DCA / forest figures."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FormatStrFormatter
from sklearn.metrics import roc_auc_score, roc_curve
import config as cfg
from utils import load_data, save, make_outcome, numeric, develop_model, mice_fit, calib, net_benefit
import statsmodels.api as sm

df = load_data(); y = make_outcome(df); CORE = cfg.PREDICTORS; X0 = numeric(df, CORE); m = cfg.M_IMPUTATIONS
res = develop_model(X0, y, B=0); pbar = res["pbar"]; n = len(y)

# --- sensitivity / specificity / PPV / NPV at fixed thresholds (bootstrap CIs) ---
def thr_metrics(t, yy, pp):
    d = (pp >= t).astype(int)
    tp = ((d == 1) & (yy == 1)).sum(); tn = ((d == 0) & (yy == 0)).sum()
    fp = ((d == 1) & (yy == 0)).sum(); fn = ((d == 0) & (yy == 1)).sum()
    return np.array([tp/(tp+fn), tn/(tn+fp), tp/(tp+fp) if tp+fp else np.nan, tn/(tn+fn)])

rng = np.random.default_rng(0); rows = []
for t in cfg.THRESHOLDS:
    est = thr_metrics(t, y, pbar)
    bs = np.array([thr_metrics(t, y[i], pbar[i]) for i in (rng.choice(n, n, True) for _ in range(cfg.N_BOOT_CI))])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    rows.append([t] + [f"{est[k]:.2f} ({lo[k]:.2f}-{hi[k]:.2f})" for k in range(4)])
thr = pd.DataFrame(rows, columns=["threshold", "Sensitivity", "Specificity", "PPV", "NPV"])
save(thr, "thresholds.csv"); print(thr.to_string(index=False))

# --- decision-curve analysis ---
ts = np.round(np.arange(0.05, 0.61, 0.01), 2); prev = y.mean()
dca = pd.DataFrame({"threshold": ts, "nb_model": net_benefit(y, pbar, ts).round(4),
                    "nb_all": (prev - (1 - prev) * ts / (1 - ts)).round(4), "nb_none": 0.0})
save(dca, "dca.csv")
win = dca[(dca.nb_model > dca.nb_all) & (dca.nb_model > 0)]
if len(win): print(f"\nModel beats treat-all and treat-none for thresholds {win.threshold.min():.2f}-{win.threshold.max():.2f}")

# --- temporal validation: develop on early years, freeze, test on later years (no leakage) ---
p_tr = p_va = None
if cfg.YEAR_COL and cfg.YEAR_COL in df:
    yr = df[cfg.YEAR_COL].to_numpy(); dev, val = yr <= cfg.TEMPORAL_CUT, yr > cfg.TEMPORAL_CUT
    Xdev, Xval, ytr, yva = X0[dev].to_numpy(), X0[val].to_numpy(), y[dev], y[val]
    ptr, pva = [], []
    for i in range(m):
        from sklearn.experimental import enable_iterative_imputer  # noqa
        from sklearn.impute import IterativeImputer
        imp_i = IterativeImputer(sample_posterior=True, random_state=i, max_iter=10)
        Xd = imp_i.fit_transform(np.column_stack([Xdev, ytr.astype(float)]))[:, :len(CORE)]      # outcome in
        ri = sm.Logit(ytr, sm.add_constant(Xd)).fit(disp=0, maxiter=200)                          # frozen on dev
        Xv = imp_i.transform(np.column_stack([Xval, np.full(val.sum(), np.nan)]))[:, :len(CORE)]  # outcome unknown
        ptr.append(np.asarray(ri.predict(sm.add_constant(Xd)))); pva.append(np.asarray(ri.predict(sm.add_constant(Xv))))
    p_tr, p_va = np.mean(ptr, 0), np.mean(pva, 0)

    def perf(yy, pp):
        citl, slope = calib(yy, pp)
        return [round(roc_auc_score(yy, pp), 3), round(np.mean((pp - yy)**2), 3), round(slope, 3), round(citl, 3)]
    mm = res["metrics"].set_index("metric")
    comp = pd.DataFrame({"metric": ["AUC", "Brier", "Calibration slope", "CITL"],
                         "all_data_apparent": [mm.loc[k, "apparent"] for k in ["AUC", "Brier score", "Calibration slope", "Calibration-in-the-large"]],
                         "temporal_dev": perf(ytr, p_tr), "temporal_val": perf(yva, p_va)})
    save(comp, "performance_comparison.csv")
    print(f"\nTemporal split: dev n={dev.sum()} ({ytr.sum()} events) | val n={val.sum()} ({yva.sum()} events)")
    print(comp.to_string(index=False))
else:
    print("\nNo year column - temporal validation skipped.")

# --- figures ---
def cal_bins(yy, pp, g):
    d = pd.DataFrame({"y": yy, "p": pp}); d["b"] = pd.qcut(d.p, g, duplicates="drop")
    return d.groupby("b", observed=True).agg(pred=("p", "mean"), obs=("y", "mean")).reset_index(drop=True)

if p_tr is not None:
    sets = [(p_tr, ytr, "Development", "#1f77b4"), (p_va, yva, "Temporal validation", "#d62728")]
else:
    sets = [(pbar, y, "All data (apparent)", "#1f77b4")]

plt.figure(figsize=(4.6, 4.6))
for pp, yy, lab, col in sets:
    f, t, _ = roc_curve(yy, pp); plt.plot(f, t, lw=2, color=col, label=f"{lab}: AUC {roc_auc_score(yy, pp):.3f}")
plt.plot([0, 1], [0, 1], "k--", lw=1); plt.xlabel("1 - Specificity"); plt.ylabel("Sensitivity"); plt.legend(loc="lower right", fontsize=9)
plt.tight_layout(); plt.savefig(f"{cfg.OUT_DIR}/figures/roc.png", dpi=150); plt.close()

plt.figure(figsize=(4.6, 4.6)); plt.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal")
for pp, yy, lab, col in sets:
    cb = cal_bins(yy, pp, 10 if len(yy) > 300 else 5); plt.plot(cb.pred, cb.obs, "o-", color=col, label=lab)
plt.xlabel("Predicted risk"); plt.ylabel("Observed frequency"); plt.legend(loc="upper left", fontsize=9)
plt.tight_layout(); plt.savefig(f"{cfg.OUT_DIR}/figures/calibration.png", dpi=150); plt.close()

plt.figure(figsize=(5.2, 4.2))
for pp, yy, lab, col in sets:
    pv = yy.mean(); plt.plot(ts, net_benefit(yy, pp, ts), lw=2, color=col, label=f"Model, {lab}")
    plt.plot(ts, pv - (1 - pv) * ts / (1 - ts), lw=1, ls="--", color=col, label=f"Treat all, {lab}")
plt.axhline(0, color="k", lw=1, label="Treat none"); plt.ylim(-0.10, y.mean() + 0.05)
plt.xlabel("Threshold probability"); plt.ylabel("Net benefit"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{cfg.OUT_DIR}/figures/dca.png", dpi=150); plt.close()

cf = res["coef"]; cf = cf[cf.term != "intercept"].iloc[::-1].reset_index(drop=True)
cols = ["#d62728" if p < 0.05 else "#999999" for p in cf.p]
plt.figure(figsize=(6, 3.2))
for i, r in cf.iterrows():
    plt.plot([r.OR_lo, r.OR_hi], [i, i], color=cols[i], lw=2); plt.plot(r.OR_per_unit, i, "s", color=cols[i], ms=8)
plt.axvline(1, color="k", ls="--", lw=1); plt.xscale("log")
plt.yticks(range(len(cf)), [cfg.LABELS.get(t, t) for t in cf.term]); ax = plt.gca()
ax.xaxis.set_minor_locator(NullLocator()); plt.xlabel("Odds ratio (95% CI, log scale)")
plt.tight_layout(); plt.savefig(f"{cfg.OUT_DIR}/figures/forest.png", dpi=150); plt.close()
print("\nFigures saved to", f"{cfg.OUT_DIR}/figures/")
