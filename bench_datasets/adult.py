"""Adult Census Income 

Kaggle: https://www.kaggle.com/datasets/uciml/adult-census-income  (adult.csv)
"""
import numpy as np
import pandas as pd

from .base import DatasetConfig

UCI_NAMES = ["age", "workclass", "fnlwgt", "education", "education.num",
             "marital.status", "occupation", "relationship", "race", "sex",
             "capital.gain", "capital.loss", "hours.per.week", "native.country", "income"]

GT_COLUMNS = {"E1_fnlwgt_notinfamily": "fnlwgt",
              "E2_age_us": "age",
              "E3_workclass_nevermarried": "workclass",
              "E4_education_female": "education"}


def load(path):
    df = pd.read_csv(path, skipinitialspace=True)
    if "income" not in df.columns and df.shape[1] == 15:      # header-less UCI file
        df = pd.read_csv(path, header=None, names=UCI_NAMES, skipinitialspace=True)
    df.columns = [str(c).strip().replace("-", ".").replace("_", ".") for c in df.columns]
    df = df.replace({"?": np.nan, " ?": np.nan})
    df["income"] = df["income"].astype(str).str.contains(">50K").astype(int)
    for c in ("capital.gain", "capital.loss"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("age", "fnlwgt", "education.num", "hours.per.week"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df.reset_index(drop=True)


def inject(df, seed=42):
    rng = np.random.default_rng(seed)
    df = df.copy().reset_index(drop=True)
    gt = {}

    # E1 additive shift: fnlwgt +100000 for Not-in-family          [numerical shift]
    m1 = (df["relationship"] == "Not-in-family") & df["fnlwgt"].notna()
    df.loc[m1, "fnlwgt"] = df.loc[m1, "fnlwgt"] + 100000.0
    gt["E1_fnlwgt_notinfamily"] = sorted(int(i) for i in np.where(m1)[0])

    # E2 multiplicative: age x0.5 for United-States                 [numerical shift]
    m2 = (df["native.country"] == "United-States") & df["age"].notna()
    df.loc[m2, "age"] = df.loc[m2, "age"] * 0.5
    gt["E2_age_us"] = sorted(int(i) for i in np.where(m2)[0])

    # E3 overwrite to a MINORITY value: workclass := Self-emp-inc
    #    for Never-married                                          [categorical shift]
    m3 = (df["marital.status"] == "Never-married") & df["workclass"].notna()
    touched = m3 & (df["workclass"] != "Self-emp-inc")
    df.loc[m3, "workclass"] = "Self-emp-inc"
    gt["E3_workclass_nevermarried"] = sorted(int(i) for i in np.where(touched)[0])

    # E4 conditional missingness: 70% of education -> NaN where sex=Female
    #    (education has no natural NaN, so the clean-reference precondition holds)
    m4 = (df["sex"] == "Female") & df["education"].notna()
    rows4 = np.where(m4)[0]
    chosen = rng.choice(rows4, size=int(0.70 * len(rows4)), replace=False)
    df.loc[chosen, "education"] = np.nan
    gt["E4_education_female"] = sorted(int(i) for i in chosen)

    return df, gt


def analyze(cleaned, clean_t, dirty_t, gt, o2p):
    out = []
    for name, rows_full in gt.items():
        col = GT_COLUMNS[name]
        rows = [o2p[i] for i in rows_full if i in o2p]
        if not rows:
            continue
        cl, dr, cn = clean_t[col], dirty_t[col], cleaned[col]
        changed = sum(1 for r in rows
                      if not ((pd.isna(cn.iat[r]) and pd.isna(dr.iat[r])) or str(cn.iat[r]) == str(dr.iat[r])))
        exact = sum(1 for r in rows
                    if (pd.isna(cn.iat[r]) and pd.isna(cl.iat[r])) or str(cn.iat[r]) == str(cl.iat[r]))
        out.append(f"{name} ({col}), {len(rows)} rows: changed {changed}/{len(rows)}, "
                   f"exactly restored {exact}/{len(rows)}")
    return out


CONFIG = DatasetConfig(
    name="adult",
    raw_path="data/adult.csv",
    target="income",
    description=("US Census income dataset. Each row is a person. Columns: age, workclass, "
                 "fnlwgt (census weight), education, education.num (years of education), "
                 "marital.status, occupation, relationship, race, sex, capital.gain, "
                 "capital.loss, hours.per.week, native.country, income (target, 1 if >50K)."),
    hints={"none": "", "weak": "", "strong": ""},          # confirmatory runs are hint-free
    numeric_features=["age", "fnlwgt", "education.num", "capital.gain", "capital.loss",
                      "hours.per.week"],
    categorical_features=["workclass", "education", "marital.status", "occupation",
                          "relationship", "race", "sex", "native.country"],
    id_cols=[],
    load=load, inject=inject, analyze=analyze,
)
