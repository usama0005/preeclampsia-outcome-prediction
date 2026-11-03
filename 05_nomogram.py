"""Step 5 - points-based nomogram built from the pooled model coefficients."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import config as cfg
from utils import load_data, make_outcome, numeric, develop_model

df = load_data(); y = make_outcome(df); CORE = cfg.PREDICTORS
coef = develop_model(numeric(df, CORE), y, B=0)["coef"].set_index("term")
b0 = coef.loc["intercept", "beta"]; bco = {c: coef.loc[c, "beta"] for c in CORE}
Xn = numeric(df, CORE)
lo = {c: np.nanpercentile(Xn[c], 1) for c in CORE}; hi = {c: np.nanpercentile(Xn[c], 99) for c in CORE}  # clip to observed range
D = max(abs(bco[c] * (hi[c] - lo[c])) for c in CORE)                       # largest effect spans 100 points
base = sum(min(bco[c] * lo[c], bco[c] * hi[c]) for c in CORE)
pts = lambda c, x: 100 * (bco[c] * x - min(bco[c] * lo[c], bco[c] * hi[c])) / D
maxpts = {c: pts(c, hi[c] if bco[c] > 0 else lo[c]) for c in CORE}; total_max = sum(maxpts.values())

# self-check: nomogram points -> risk must equal the model's predicted risk
chk = Xn.dropna().iloc[:20]
nomo = 1 / (1 + np.exp(-(b0 + base + chk.apply(lambda r: sum(pts(c, r[c]) for c in CORE), axis=1) * D / 100)))
modl = 1 / (1 + np.exp(-(b0 + chk.mul(pd.Series(bco)).sum(axis=1))))
assert (nomo - modl).abs().max() < 1e-9, "nomogram does not reproduce model risk"

order = sorted(CORE, key=lambda c: -maxpts[c])
fig, ax = plt.subplots(figsize=(9, 6)); ax.axis("off")
def ruler(yrow, x0, x1, label): ax.plot([x0, x1], [yrow, yrow], "k-", lw=1); ax.text(-3, yrow, label, ha="right", va="center", fontsize=9)
ruler(9, 0, 100, "Points")
for p in range(0, 101, 10): ax.plot([p, p], [9, 9.15], "k-", lw=1); ax.text(p, 9.3, str(p), ha="center", fontsize=7)
for i, c in enumerate(order):
    yrow = 8 - i; pl, ph = pts(c, lo[c]), pts(c, hi[c]); ruler(yrow, min(pl, ph), max(pl, ph), cfg.LABELS.get(c, c))
    for tv in np.linspace(lo[c], hi[c], 5):
        ax.plot([pts(c, tv)] * 2, [yrow, yrow + 0.12], "k-", lw=1); ax.text(pts(c, tv), yrow + 0.26, f"{tv:.1f}", ha="center", fontsize=7)
yt = 8 - len(order) - 0.5; ruler(yt, 0, 100, "Total points")
for tv in np.linspace(0, total_max, 6):
    ax.plot([100 * tv / total_max] * 2, [yt, yt + 0.12], "k-", lw=1); ax.text(100 * tv / total_max, yt + 0.26, f"{tv:.0f}", ha="center", fontsize=7)
yr = yt - 1; ruler(yr, 0, 100, "Predicted risk")
for risk in [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
    tp = (np.log(risk / (1 - risk)) - b0 - base) * 100 / D
    if 0 <= tp <= total_max:
        ax.plot([100 * tp / total_max] * 2, [yr, yr + 0.12], "k-", lw=1); ax.text(100 * tp / total_max, yr + 0.26, f"{risk:.2f}", ha="center", fontsize=7)
ax.set_xlim(-17, 104); ax.set_ylim(yr - 0.6, 9.7)
plt.tight_layout(); plt.savefig(f"{cfg.OUT_DIR}/figures/nomogram.png", dpi=150); plt.close()
print(f"Saved {cfg.OUT_DIR}/figures/nomogram.png | total points 0-{total_max:.0f}")
