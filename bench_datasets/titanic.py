"""Titanic - Bendinelli et al. Appendix A.1.1 errors."""
import numpy as np
import pandas as pd
from .base import DatasetConfig

HIGH_STATUS = ["Dr", "Lady", "Sir", "Countess", "the Countess", "Col", "Major",
               "Capt", "Jonkheer", "Don", "Dona", "Rev"]


def _title(s):
    return s.str.extract(r",\s*([^\.]+)\.")[0].str.strip()


def load(path):
    df = pd.read_csv(path)
    return df.reset_index(drop=True)


def inject(df, seed=42):
    rng = np.random.default_rng(seed)
    df = df.copy().reset_index(drop=True)
    gt = {}
    title = _title(df["Name"])
    # E1: 50% of female survivors (Miss/Mrs) -> Sex = male            [categorical shift]
    m1 = (df["Sex"] == "female") & (df["Survived"] == 1) & title.isin(["Miss", "Mrs"])
    idx1 = rng.choice(df[m1].index.to_numpy(), size=int(m1.sum() * 0.5), replace=False)
    df.loc[idx1, "Sex"] = "male"
    gt["error1_sex"] = sorted(int(i) for i in idx1)
    # E2: 50% of married female non-survivors (Mrs) -> Age in [2,8]  [numerical shift]
    m2 = (df["Sex"] == "female") & (df["Survived"] == 0) & (title == "Mrs")
    idx2 = rng.choice(df[m2].index.to_numpy(), size=int(m2.sum() * 0.5), replace=False)
    df.loc[idx2, "Age"] = rng.integers(2, 9, len(idx2)).astype(float)
    gt["error2_age"] = sorted(int(i) for i in idx2)
    # E3: high-status titles -> Fare * 0.10                            [numerical shift]
    idx3 = df[title.isin(HIGH_STATUS)].index.to_numpy()
    df.loc[idx3, "Fare"] = df.loc[idx3, "Fare"] * 0.10
    gt["error3_fare"] = sorted(int(i) for i in idx3)
    return df, gt


def analyze(cleaned, clean_t, dirty_t, gt, o2p):
    def eq(a, b):
        if pd.isna(a) and pd.isna(b): return True
        try: return abs(float(a) - float(b)) < 1e-6
        except (TypeError, ValueError): return str(a) == str(b)
    out = []
    e1 = [o2p[i] for i in gt["error1_sex"] if i in o2p]
    f1 = sum(1 for p in e1 if str(cleaned.loc[p, "Sex"]) == "female")
    out.append(f"ERROR 1 (Sex->male), {len(e1)} rows: restored to female {f1}/{len(e1)}")
    e2 = [o2p[i] for i in gt["error2_age"] if i in o2p]
    f2 = sum(1 for p in e2 if not (2 <= float(cleaned.loc[p, "Age"]) <= 8) if pd.notna(cleaned.loc[p, "Age"])) \
        + sum(1 for p in e2 if pd.isna(cleaned.loc[p, "Age"]))
    out.append(f"ERROR 2 (Age->2-8), {len(e2)} rows: no longer 2-8 {f2}/{len(e2)}")
    e3 = [o2p[i] for i in gt["error3_fare"] if i in o2p
          and not eq(clean_t.loc[o2p[i], "Fare"], dirty_t.loc[o2p[i], "Fare"])]
    f3 = sum(1 for p in e3 if not eq(cleaned.loc[p, "Fare"], dirty_t.loc[p, "Fare"]))
    out.append(f"ERROR 3 (Fare x0.10), {len(e3)} rows: changed from corrupted {f3}/{len(e3)}")
    return out


CONFIG = DatasetConfig(
    name="titanic",
    raw_path="data/titanic.csv",
    target="Survived",
    description=("Titanic passenger dataset. Each row is a passenger. Columns: PassengerId, "
                 "Survived (target, 1=survived), Pclass (1/2/3 ticket class), Name (includes a "
                 "title such as Mr/Mrs/Miss/Master/Dr), Sex, Age (years), SibSp (siblings/spouses "
                 "aboard), Parch (parents/children aboard), Ticket, Fare (ticket price), Cabin, "
                 "Embarked (port C/Q/S)."),
    hints={
        "none": "",
        "weak": "\n\nHint: Errors are in the Sex, Age and Fare columns.",
        "strong": ("\n\nHint: Errors are here: Female survivors had their sex entry corrupted. "
                   "The same happened for the age of female married non-survivors, and the fare "
                   "of some passengers with high social status was corrupted."),
    },
    numeric_features=["Pclass", "Age", "SibSp", "Parch", "Fare"],
    categorical_features=["Sex", "Embarked"],
    id_cols=["PassengerId", "Name", "Ticket", "Cabin"],
    load=load, inject=inject, analyze=analyze,
)
