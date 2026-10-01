"""Robustness and extension analyses for the subsidy -> obesity estimate.

All models are two-way (country + year) fixed effects estimated on the swept
data, with controls log GDP, log agricultural GHG and log food CPI. The DML
estimate in dml_analysis.py is close to the linear one, so the cheaper linear
model is used where many refits are needed.

  1. Wild cluster bootstrap-t (restricted, Webb weights): p-value that is
     reliable with only 13 clusters.
  2. Leave-one-country-out: linear FE and DML (GBM) dropping each country.
  3. Lead/lag profile: subsidy at t+4 ... t-8, one at a time.
  4. Non-linearity: quadratic term in lagged subsidy.
  5. Heterogeneity: effect in high- vs low-subsidy and high- vs low-income
     countries.
  6. Omitted-variable sensitivity: Cinelli & Hazlett (2020) robustness value.

Outputs go to results/.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm

from clean_panel import two_way_demean
from dml_analysis import LEARNERS, X, Y, analysis_sample, dml, p_value, sweep

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "oecd_master_panel_clean.csv"
OUT = ROOT / "results"
D = "log_ag_subsidy_lag4"
B = 9999
SEED = 20260101

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


def fe(w, d_cols, y=Y):
    """OLS on swept data; returns the fit with country-clustered SEs."""
    exog = w[d_cols + X]
    groups = pd.factorize(w["country_code"])[0]
    return sm.OLS(w[y], exog).fit(cov_type="cluster", cov_kwds={"groups": groups})


def wild_cluster_bootstrap(w, d, rng):
    """Restricted wild cluster bootstrap-t for H0: theta = 0 (Webb 6-point weights)."""
    groups = pd.factorize(w["country_code"])[0]
    G = groups.max() + 1
    xs, xd, y = w[X].to_numpy(), w[[d] + X].to_numpy(), w[Y].to_numpy()
    t_obs = fe(w, [d]).tvalues[d]
    restricted = sm.OLS(y, xs).fit()
    fitted0, u0 = restricted.fittedvalues, restricted.resid
    webb = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])
    xtx_inv = np.linalg.inv(xd.T @ xd)
    t_star = np.empty(B)
    for b in range(B):
        ystar = fitted0 + u0 * rng.choice(webb, G)[groups]
        beta = xtx_inv @ xd.T @ ystar
        e = ystar - xd @ beta
        meat = sum(np.outer(s, s) for s in
                   (xd[groups == g].T @ e[groups == g] for g in range(G)))
        n, k = xd.shape
        adj = G / (G - 1) * (n - 1) / (n - k)
        t_star[b] = beta[0] / np.sqrt(adj * (xtx_inv @ meat @ xtx_inv)[0, 0])
    return t_obs, np.mean(np.abs(t_star) >= abs(t_obs))


def robustness_value(t, dof, q=1.0):
    """Cinelli & Hazlett RV_q: confounder strength (partial R2 with both D and Y)
    needed to reduce the estimate by 100*q percent."""
    f = q * abs(t) / np.sqrt(dof)
    return 0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2), t ** 2 / (t ** 2 + dof)


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATA).sort_values(["country_code", "year"])
    base = analysis_sample(df)
    w = sweep(base, [Y, D] + X)
    rows = []

    # 1. Wild cluster bootstrap
    main_fit = fe(w, [D])
    t_obs, p_wcb = wild_cluster_bootstrap(w, D, np.random.default_rng(SEED))
    rows.append({"analysis": "Main linear FE", "detail": "cluster-robust",
                 "theta": main_fit.params[D], "se": main_fit.bse[D],
                 "p": main_fit.pvalues[D], "N": len(w)})
    rows.append({"analysis": "Main linear FE", "detail": f"wild cluster bootstrap ({B} draws)",
                 "theta": main_fit.params[D], "se": np.nan, "p": p_wcb, "N": len(w)})
    print(f"wild cluster bootstrap: t={t_obs:.2f}, p={p_wcb:.3f}")

    # 6. Sensitivity (classical OLS t on the swept data; dof absorbs the FE)
    n_fe = base["country_code"].nunique() + base["year"].nunique() - 1
    dof = len(w) - n_fe - len(X) - 1
    t_cls = sm.OLS(w[Y], w[[D] + X]).fit().tvalues[D] * np.sqrt(dof / (len(w) - len(X) - 1))
    rv, r2 = robustness_value(t_cls, dof)
    rv_sig, _ = robustness_value(t_cls - 1.96 * np.sign(t_cls), dof)
    rows.append({"analysis": "Sensitivity (Cinelli-Hazlett)", "detail":
                 f"partial R2 of treatment={r2:.3f}; RV(theta=0)={rv:.3f}; RV(p<.05)={rv_sig:.3f}",
                 "theta": np.nan, "se": np.nan, "p": np.nan, "N": len(w)})
    print(f"sensitivity: partial R2={r2:.3f}, RV={rv:.3f}, RV_alpha={rv_sig:.3f}")

    # 2. Leave-one-country-out
    loco = []
    for c in sorted(base["country_code"].unique()):
        s = base[base["country_code"] != c].reset_index(drop=True)
        ws = sweep(s, [Y, D] + X)
        r = fe(ws, [D])
        th_ml, _, se_ml = dml(ws, Y, D, LEARNERS["gbm"], 3, SEED)
        loco.append({"dropped": c, "theta_fe": r.params[D], "se_fe": r.bse[D], "p_fe": r.pvalues[D],
                     "theta_dml": th_ml, "se_dml": se_ml, "p_dml": p_value(th_ml, se_ml)})
    loco = pd.DataFrame(loco)
    loco.to_csv(OUT / "robustness_leave_one_out.csv", index=False)
    print(loco.round(3).to_string(index=False))

    # 3. Lead / lag profile built from the in-sample subsidy series
    prof = []
    for k in range(-4, 9):
        col = f"sub_k{k}"
        df[col] = df.groupby("country_code")["log_ag_subsidy_pse"].shift(k)
        s = analysis_sample(df).dropna(subset=[col]).reset_index(drop=True)
        ws = sweep(s, [Y, col] + X)
        r = fe(ws, [col])
        prof.append({"k": k, "theta": r.params[col], "se": r.bse[col], "p": r.pvalues[col],
                     "N": len(ws)})
    prof = pd.DataFrame(prof)
    prof.to_csv(OUT / "robustness_lag_profile.csv", index=False)
    print(prof.round(3).to_string(index=False))

    # 4. Non-linearity
    b2 = base.copy()
    b2["d_c"] = b2[D] - b2[D].mean()
    b2["d_sq"] = b2["d_c"] ** 2
    ws = sweep(b2, [Y, "d_c", "d_sq"] + X)
    r = fe(ws, ["d_c", "d_sq"])
    rows.append({"analysis": "Quadratic", "detail": "linear term (at mean subsidy)",
                 "theta": r.params["d_c"], "se": r.bse["d_c"], "p": r.pvalues["d_c"], "N": len(ws)})
    rows.append({"analysis": "Quadratic", "detail": "squared term",
                 "theta": r.params["d_sq"], "se": r.bse["d_sq"], "p": r.pvalues["d_sq"], "N": len(ws)})

    # 5. Heterogeneity by time-invariant country traits (median split)
    means = base.groupby("country_code")[["log_ag_subsidy_pse", "log_gdp_per_capita"]].mean()
    for trait, label in [("log_ag_subsidy_pse", "high-subsidy"), ("log_gdp_per_capita", "high-income")]:
        hi = means.index[means[trait] > means[trait].median()]
        b3 = base.copy()
        b3["d_x_hi"] = b3[D] * b3["country_code"].isin(hi)
        ws = sweep(b3, [Y, D, "d_x_hi"] + X)
        r = fe(ws, [D, "d_x_hi"])
        rows.append({"analysis": f"Heterogeneity: {label}", "detail":
                     f"effect in other countries; {label} = {', '.join(hi)}",
                     "theta": r.params[D], "se": r.bse[D], "p": r.pvalues[D], "N": len(ws)})
        rows.append({"analysis": f"Heterogeneity: {label}", "detail": f"difference for {label}",
                     "theta": r.params["d_x_hi"], "se": r.bse["d_x_hi"],
                     "p": r.pvalues["d_x_hi"], "N": len(ws)})

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "robustness_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))

    plot(prof, loco, main_fit.params[D])


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.axhline(0, color=MUTED, lw=1)


def plot(prof, loco, theta_main):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor="#fcfcfb")
    ax = axes[0]
    style(ax)
    lead = prof["k"] < 0
    for mask, color, label in [(lead, ORANGE, "leads (placebo)"), (~lead, BLUE, "lags")]:
        p = prof[mask]
        ax.errorbar(p["k"], p["theta"], yerr=1.96 * p["se"], fmt="o", ms=8, color=color,
                    ecolor=color, elinewidth=2, capsize=0, label=label)
    ax.axvline(-0.5, color=GRID, lw=1, ls="--")
    ax.set(title="Effect of subsidy k years before the obesity measurement",
           xlabel="k (negative = years ahead)", ylabel="θ (pp obesity per log-unit subsidy)")
    ax.set_xticks(prof["k"])
    ax.legend(frameon=False, labelcolor=INK)

    ax = axes[1]
    style(ax)
    xs = np.arange(len(loco))
    ax.errorbar(xs - 0.15, loco["theta_fe"], yerr=1.96 * loco["se_fe"], fmt="o", ms=8,
                color=BLUE, elinewidth=2, capsize=0, label="linear FE")
    ax.errorbar(xs + 0.15, loco["theta_dml"], yerr=1.96 * loco["se_dml"], fmt="s", ms=8,
                color=ORANGE, elinewidth=2, capsize=0, label="DML (GBM)")
    ax.axhline(theta_main, color=MUTED, lw=1, ls=":")
    ax.set_xticks(xs, loco["dropped"])
    ax.set(title="Leave-one-country-out estimates (95% clustered CI)",
           xlabel="country dropped", ylabel="θ")
    ax.legend(frameon=False, labelcolor=INK)
    for a in axes:
        a.set_facecolor("#fcfcfb")
        a.title.set_color(INK)
    fig.tight_layout()
    fig.savefig(OUT / "robustness.png", dpi=150)


if __name__ == "__main__":
    main()
