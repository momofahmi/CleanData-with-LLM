"""Meat Consumption (Our World in Data / Kaggle 'per capita meat consumption by type').
Bendinelli et al. Appendix A.1.2 errors. 
"""
import numpy as np
import pandas as pd
from .base import DatasetConfig

LANDLOCKED = ["Afghanistan", "Burkina Faso", "Chad", "Burundi", "Central African Republic",
              "Niger", "Nepal", "Mali", "Tajikistan", "Uzbekistan", "Kyrgyzstan"]
GROWTH_COUNTRIES = ["Mauritius", "Italy", "Japan", "Vietnam", "China", "Mexico"]
POULTRY_YEARS = [1986, 1990, 1993, 1995, 2000, 2005, 2010, 2015]
MEAT_KEYS = {"Poultry": ["poultry"], "Beef": ["beef", "bovine"], "Pork": ["pig", "pork"],
             "Sheep": ["sheep", "mutton", "goat"], "Fish": ["fish", "seafood"], "Other": ["other"]}
MEAT_COLS = list(MEAT_KEYS)


def load(path):
    raw = pd.read_csv(path)
    cols = {c.lower(): c for c in raw.columns}
    def find(*keys):
        for k in keys:
            for lc, orig in cols.items():
                if k in lc:
                    return orig
        return None
    ent = find("entity", "country", "area")
    code = find("code")
    year = find("year", "time")
    out = pd.DataFrame({"Entity": raw[ent].astype(str), "Year": pd.to_numeric(raw[year], errors="coerce")})
    if code:
        out["Code"] = raw[code]
    for canon, keys in MEAT_KEYS.items():
        c = find(*keys)
        if c is not None:
            out[canon] = pd.to_numeric(raw[c], errors="coerce")
    present = [c for c in MEAT_COLS if c in out.columns]
    if "Code" in out.columns:   # drop aggregates (regions have no ISO code / OWID_ codes)
        out = out[out["Code"].notna() & ~out["Code"].astype(str).str.startswith("OWID")]
    out = out.dropna(subset=["Year"]).copy()
    out["Year"] = out["Year"].astype(int)
    out = out.dropna(subset=present, how="all")
    out["Total"] = out[present].fillna(0).sum(axis=1)
    med = out.groupby("Year")["Total"].transform("median")
    out["HighConsumption"] = (out["Total"] > med).astype(int)
    out = out.drop(columns=["Total"]).sort_values(["Entity", "Year"]).reset_index(drop=True)
    return out


def inject(df, seed=42):
    rng = np.random.default_rng(seed)
    df = df.copy().reset_index(drop=True)
    present = [c for c in MEAT_COLS if c in df.columns]
    gt = {}
    # E1: Poultry near zero for specific years                       [year-level collapse]
    idx1 = df[df["Year"].isin(POULTRY_YEARS)].index.to_numpy()
    df.loc[idx1, "Poultry"] = df.loc[idx1, "Poultry"] * rng.uniform(0.0, 0.03, len(idx1))
    gt["error1_poultry_years"] = sorted(int(i) for i in idx1)
    # E2: Fish -> 85th-95th percentile for landlocked countries      [entity-level overwrite]
    q85, q95 = df["Fish"].quantile(0.85), df["Fish"].quantile(0.95)
    idx2 = df[df["Entity"].isin(LANDLOCKED)].index.to_numpy()
    df.loc[idx2, "Fish"] = rng.uniform(q85, q95, len(idx2))
    gt["error2_fish_landlocked"] = sorted(int(i) for i in idx2)
    # E3: total meat +30% per year for six countries, 1997-2004      [multiplicative shift]
    m3 = df["Entity"].isin(GROWTH_COUNTRIES) & df["Year"].between(1997, 2004)
    idx3 = df[m3].index.to_numpy()
    df.loc[idx3, present] = df.loc[idx3, present] * 1.30
    gt["error3_growth_1997_2004"] = sorted(int(i) for i in idx3)
    return df, gt


def analyze(cleaned, clean_t, dirty_t, gt, o2p):
    def close(a, b, tol=0.05):
        try:
            a, b = float(a), float(b)
            return abs(a - b) <= tol * max(abs(b), 1e-9)
        except (TypeError, ValueError):
            return False
    out = []
    e1 = [o2p[i] for i in gt["error1_poultry_years"] if i in o2p and float(clean_t.loc[o2p[i], "Poultry"] or 0) > 0]
    ch1 = sum(1 for p in e1 if not close(cleaned.loc[p, "Poultry"], dirty_t.loc[p, "Poultry"], 0.01))
    ex1 = sum(1 for p in e1 if close(cleaned.loc[p, "Poultry"], clean_t.loc[p, "Poultry"], 0.15))
    out.append(f"ERROR 1 (Poultry ~0 in {POULTRY_YEARS}), {len(e1)} rows: changed {ch1}/{len(e1)} | within 15% of truth {ex1}/{len(e1)}")
    e2 = [o2p[i] for i in gt["error2_fish_landlocked"] if i in o2p]
    ch2 = sum(1 for p in e2 if not close(cleaned.loc[p, "Fish"], dirty_t.loc[p, "Fish"], 0.01))
    ex2 = sum(1 for p in e2 if close(cleaned.loc[p, "Fish"], clean_t.loc[p, "Fish"], 0.25))
    out.append(f"ERROR 2 (Fish inflated, landlocked), {len(e2)} rows: changed {ch2}/{len(e2)} | within 25% of truth {ex2}/{len(e2)}")
    e3 = [o2p[i] for i in gt["error3_growth_1997_2004"] if i in o2p]
    ch3 = sum(1 for p in e3 if not close(cleaned.loc[p, "Beef"] if "Beef" in cleaned else cleaned.loc[p, "Poultry"],
                                         dirty_t.loc[p, "Beef"] if "Beef" in dirty_t else dirty_t.loc[p, "Poultry"], 0.01))
    ex3 = sum(1 for p in e3 if close(cleaned.loc[p, "Beef"] if "Beef" in cleaned else cleaned.loc[p, "Poultry"],
                                     clean_t.loc[p, "Beef"] if "Beef" in clean_t else clean_t.loc[p, "Poultry"], 0.05))
    out.append(f"ERROR 3 (x1.30, 6 countries 1997-2004), {len(e3)} rows: changed {ch3}/{len(e3)} | within 5% of truth {ex3}/{len(e3)}")
    return out


CONFIG = DatasetConfig(
    name="meat",
    raw_path="data/meat_consumption.csv",
    target="HighConsumption",
    description=("Per-capita meat consumption by country and year (kg per person per year). "
                 "Each row is one country in one year. Columns: Entity (country name), Code (ISO code), "
                 "Year, Poultry, Beef, Pork, Sheep, Fish, Other (consumption by meat type), and "
                 "HighConsumption (target: 1 if the country's total meat consumption that year is above "
                 "the median across countries for that year)."),
    hints={   
        "none": "",
        "weak": "\n\nHint: Errors are in the Poultry and Fish columns, and in the overall consumption of a few countries.",
        "strong": ("\n\nHint: Errors are here: poultry consumption is wrongly near zero for some years; fish "
                   "consumption is inflated for landlocked countries; and the total consumption of a few "
                   "countries between 1997 and 2004 is inflated."),
    },
    numeric_features=["Year", "Poultry", "Beef", "Pork", "Sheep", "Fish", "Other"],
    categorical_features=[],
    id_cols=["Entity", "Code"],
    entity_col="Entity", time_col="Year",
    load=load, inject=inject, analyze=analyze,
)
