"""Clean the OECD obesity/agricultural-subsidy panel.

Fixes applied to data/oecd_master_panel_analysis_ready_2.csv:
  1. obesity_rate was carried forward between survey years. Interior gaps are
     now linearly interpolated between survey years; years after a country's
     last survey keep the carried value but are flagged (obesity_carried=1).
     obesity_survey_year=1 marks rows that hold an actual survey value.
  2. Turkey's 2022-24 hyperinflation makes food_cpi extreme. Added log_food_cpi
     and a tur_hyperinflation flag for exclusion in robustness checks.
  3. GBR (only 4 years) is dropped: it adds almost no within-country variation.
  4. Added log versions of the skewed level variables so coefficients read as
     elasticities / semi-elasticities.
  5. *_demeaned columns are recomputed as exact two-way (country + year)
     within-transformations on the final sample (alternating projections,
     which is correct for unbalanced panels).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "oecd_master_panel_analysis_ready_2.csv"
OUT = ROOT / "data" / "oecd_master_panel_clean.csv"

LEVELS = ["obesity_rate", "ag_subsidy_pse", "gdp_per_capita",
          "agri_ghg_emissions", "food_cpi", "ag_subsidy_lag4"]
LOGGED = ["ag_subsidy_pse", "ag_subsidy_lag4", "gdp_per_capita",
          "agri_ghg_emissions", "food_cpi"]


def fix_obesity(g):
    g = g.sort_values("year").copy()
    ob = g["obesity_rate"]
    survey = ob.ne(ob.shift())  # first year and every change = a survey value
    last_survey = g.loc[survey, "year"].max()
    g["obesity_survey_year"] = survey.astype(int)
    g["obesity_carried"] = (g["year"] > last_survey).astype(int)
    interp = ob.where(survey).set_axis(g["year"]).interpolate(
        method="index", limit_area="inside")
    g["obesity_rate"] = interp.to_numpy()
    g.loc[g["obesity_carried"] == 1, "obesity_rate"] = ob[g["obesity_carried"] == 1]
    return g


def two_way_demean(df, col, tol=1e-10, max_iter=1000):
    x = df[col].astype(float).to_numpy()
    for _ in range(max_iter):
        prev = x.copy()
        x = x - pd.Series(x, index=df.index).groupby(df["country_code"]).transform("mean").to_numpy()
        x = x - pd.Series(x, index=df.index).groupby(df["year"]).transform("mean").to_numpy()
        if np.abs(x - prev).max() < tol:
            break
    return x


def main():
    df = pd.read_csv(SRC)
    df = df.drop(columns=[c for c in df.columns if c.endswith("_demeaned")])
    df = df[df["country_code"] != "GBR"]
    df = pd.concat([fix_obesity(g) for _, g in df.groupby("country_code")],
                   ignore_index=True)
    df["tur_hyperinflation"] = ((df["country_code"] == "TUR") & (df["year"] >= 2022)).astype(int)
    for c in LOGGED:
        df[f"log_{c}"] = np.log(df[c])
    for c in LEVELS + [f"log_{c}" for c in LOGGED]:
        df[f"{c}_demeaned"] = two_way_demean(df, c)
    df = df.sort_values(["country_code", "year"])
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}: {df.shape[0]} rows x {df.shape[1]} cols")


if __name__ == "__main__":
    main()
