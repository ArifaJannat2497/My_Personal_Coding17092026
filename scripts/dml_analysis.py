"""Double/debiased machine learning (partially linear model) on the OECD panel.

Pipeline
  1. Panel + lag: treatment D = log agricultural subsidy (PSE) lagged 4 years.
  2. Within-transformation: Y, D and X are demeaned by country and year
     (exact two-way FWL sweep on the analysis sample).
  3. Cross-fitting: K folds, assigned by country so no country is in both the
     training and the prediction fold; repeated over several random splits.
  4. Nuisance models: gradient boosting and random forest for
     g(X) = E[Y | X] and m(X) = E[D | X].
  5. Neyman-orthogonal residuals: Y - g_hat(X), D - m_hat(X).
  6. theta from OLS of Y-residual on D-residual, HC3 standard errors
     (country-clustered SEs reported alongside because of serial correlation).
  7. Validation: permutation feature importance of the nuisance models and
     placebo tests (future treatment, within-country shuffled treatment,
     random-walk outcome).

Outputs go to results/.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance

from clean_panel import two_way_demean

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "oecd_master_panel_clean.csv"
OUT = ROOT / "results"

Y = "obesity_rate"
D = "log_ag_subsidy_lag4"
X = ["log_gdp_per_capita", "log_agri_ghg_emissions", "log_food_cpi"]
K = 5
N_SPLITS = 20       # repeated cross-fitting draws; theta is the median
N_PERM = 200        # placebo permutations
SEED = 20260101

LEARNERS = {
    "gbm": GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05,
                                     subsample=0.8, random_state=0),
    "rf": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1),
}


def analysis_sample(df, survey_only=False):
    s = df[(df["obesity_carried"] == 0) & (df["tur_hyperinflation"] == 0)]
    if survey_only:
        s = s[s["obesity_survey_year"] == 1]
    return s.reset_index(drop=True)


def sweep(df, cols):
    """Two-way (country, year) within-transformation of every column in cols."""
    out = df[["country_code", "year"]].copy()
    for c in cols:
        out[c] = two_way_demean(df, c)
    return out


def country_folds(countries, rng):
    uniq = rng.permutation(countries.unique())
    fold_of = {c: i % K for i, c in enumerate(uniq)}
    return countries.map(fold_of).to_numpy()


def cross_fit(w, y, d, learner, rng):
    """Out-of-fold residuals of y and d on the swept covariates."""
    folds = country_folds(w["country_code"], rng)
    xs = w[X].to_numpy()
    ry, rd = np.empty(len(w)), np.empty(len(w))
    for k in range(K):
        tr, te = folds != k, folds == k
        ry[te] = y[te] - clone(learner).fit(xs[tr], y[tr]).predict(xs[te])
        rd[te] = d[te] - clone(learner).fit(xs[tr], d[tr]).predict(xs[te])
    return ry, rd


def final_stage(ry, rd, groups):
    exog = sm.add_constant(rd)
    hc3 = sm.OLS(ry, exog).fit(cov_type="HC3")
    cl = sm.OLS(ry, exog).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]})
    return hc3.params[1], hc3.bse[1], hc3.pvalues[1], cl.bse[1], cl.pvalues[1]


def dml(w, y_col, d_col, learner, n_splits, seed):
    """Repeated cross-fitted DML; median aggregation (Chernozhukov et al. 2018)."""
    rng = np.random.default_rng(seed)
    y, d = w[y_col].to_numpy(), w[d_col].to_numpy()
    draws = np.array([final_stage(*cross_fit(w, y, d, learner, rng), w["country_code"])
                      for _ in range(n_splits)])
    theta = np.median(draws[:, 0])
    # median-adjusted SE accounts for variation across splits
    se_hc3 = np.median(np.sqrt(draws[:, 1] ** 2 + (draws[:, 0] - theta) ** 2))
    se_cl = np.median(np.sqrt(draws[:, 3] ** 2 + (draws[:, 0] - theta) ** 2))
    return theta, se_hc3, se_cl


def p_value(theta, se):
    from scipy.stats import norm
    return 2 * norm.sf(abs(theta / se))


def importance(w, learner, seed):
    rng = np.random.default_rng(seed)
    folds = country_folds(w["country_code"], rng)
    xs = w[X].to_numpy()
    rows = []
    for target in (Y, D):
        t = w[target].to_numpy()
        for k in range(K):
            tr, te = folds != k, folds == k
            m = clone(learner).fit(xs[tr], t[tr])
            pi = permutation_importance(m, xs[te], t[te], n_repeats=30, random_state=seed,
                                        scoring="neg_mean_squared_error")
            for j, c in enumerate(X):
                rows.append({"nuisance": "g(X): outcome" if target == Y else "m(X): treatment",
                             "feature": c, "fold": k, "mse_increase": pi.importances_mean[j]})
    return (pd.DataFrame(rows).groupby(["nuisance", "feature"])["mse_increase"]
            .mean().reset_index())


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATA).sort_values(["country_code", "year"])
    df["log_ag_subsidy_lead4"] = df.groupby("country_code")["log_ag_subsidy_pse"].shift(-4)

    rows = []

    def record(label, sample, w, y_col, d_col, learner_name, n_splits=N_SPLITS):
        th, se3, secl = dml(w, y_col, d_col, LEARNERS[learner_name], n_splits, SEED)
        rows.append({"model": label, "sample": sample, "learner": learner_name, "N": len(w),
                     "countries": w["country_code"].nunique(), "theta": th,
                     "se_hc3": se3, "p_hc3": p_value(th, se3),
                     "se_cluster": secl, "p_cluster": p_value(th, secl)})
        print(f"{label:28s} {sample:14s} {learner_name:4s} theta={th:7.3f} "
              f"se_hc3={se3:.3f} p={p_value(th, se3):.3f} | se_cl={secl:.3f} p={p_value(th, secl):.3f}")

    # Main estimates
    for survey_only, sample in [(False, "main"), (True, "survey_years")]:
        s = analysis_sample(df, survey_only)
        w = sweep(s, [Y, D] + X)
        for name in LEARNERS:
            record("DML (lag-4 subsidy)", sample, w, Y, D, name)
        # linear two-way FE benchmark: the same FWL residualization with OLS nuisances
        lin = lambda c: sm.OLS(w[c], sm.add_constant(w[X])).fit().resid.to_numpy()
        ols = final_stage(lin(Y), lin(D), w["country_code"])
        rows.append({"model": "Linear two-way FE (FWL)", "sample": sample, "learner": "ols",
                     "N": len(w), "countries": w["country_code"].nunique(), "theta": ols[0],
                     "se_hc3": ols[1], "p_hc3": ols[2], "se_cluster": ols[3], "p_cluster": ols[4]})

    # Placebo 1: future treatment (subsidy 4 years ahead cannot cause obesity today)
    s = analysis_sample(df).dropna(subset=["log_ag_subsidy_lead4"]).reset_index(drop=True)
    w = sweep(s, [Y, "log_ag_subsidy_lead4"] + X)
    record("Placebo: lead-4 subsidy", "main", w, Y, "log_ag_subsidy_lead4", "gbm")

    # Placebo 2: random-walk outcome unrelated to subsidies
    s = analysis_sample(df).copy()
    rng = np.random.default_rng(SEED)
    s["placebo_rw"] = s.groupby("country_code")["year"].transform(
        lambda g: np.cumsum(rng.normal(size=len(g))))
    w = sweep(s, ["placebo_rw", D] + X)
    record("Placebo: random-walk outcome", "main", w, "placebo_rw", D, "gbm")

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "dml_estimates.csv", index=False)

    # Placebo 3: randomization inference, shuffling treatment within country
    s = analysis_sample(df)
    w = sweep(s, [Y, D] + X)
    theta_obs = results.query("model == 'DML (lag-4 subsidy)' and sample == 'main' "
                              "and learner == 'gbm'")["theta"].iloc[0]
    rng = np.random.default_rng(SEED + 1)
    null = []
    for i in range(N_PERM):
        sp = s.copy()
        sp[D] = sp.groupby("country_code")[D].transform(lambda g: rng.permutation(g.to_numpy()))
        wp = sweep(sp, [Y, D] + X)
        null.append(dml(wp, Y, D, LEARNERS["gbm"], 1, SEED + i)[0])
    null = np.array(null)
    p_perm = (1 + np.sum(np.abs(null) >= abs(theta_obs))) / (1 + N_PERM)
    pd.DataFrame({"theta_permuted": null}).to_csv(OUT / "placebo_permutation.csv", index=False)
    print(f"Permutation placebo: observed theta={theta_obs:.3f}, p={p_perm:.3f} ({N_PERM} draws)")

    imp = importance(w, LEARNERS["gbm"], SEED)
    imp.to_csv(OUT / "permutation_importance.csv", index=False)
    print(imp.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(null, bins=30, color="#9aa5b1")
    axes[0].axvline(theta_obs, color="#c0392b", lw=2, label=f"observed θ = {theta_obs:.2f}")
    axes[0].set(title=f"Within-country permutation placebo (p = {p_perm:.3f})",
                xlabel="θ under shuffled treatment", ylabel="count")
    axes[0].legend(frameon=False)
    piv = imp.pivot(index="feature", columns="nuisance", values="mse_increase")
    piv.plot.barh(ax=axes[1], color=["#2e86c1", "#e67e22"])
    axes[1].set(title="Permutation importance of nuisance models (GBM)",
                xlabel="increase in held-out MSE", ylabel="")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "dml_validation.png", dpi=150)

    with open(OUT / "placebo_permutation_summary.txt", "w") as f:
        f.write(f"observed_theta={theta_obs:.6f}\np_value={p_perm:.6f}\nn_perm={N_PERM}\n")


if __name__ == "__main__":
    main()
