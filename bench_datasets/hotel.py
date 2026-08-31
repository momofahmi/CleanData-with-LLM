"""Hotel Bookings - Bendinelli et al. Appendix A.1.3 errors."""
import numpy as np
import pandas as pd
from .base import DatasetConfig

YEAR = "arrival_date_year"


def load(path):
    df = pd.read_csv(path)
    return df.reset_index(drop=True)


def inject(df, seed=42):
    rng = np.random.default_rng(seed)
    df = df.copy().reset_index(drop=True)
    gt = {}
    # E1: lead_time + 10 for all 2016 records                       [numerical shift]
    idx1 = df[df[YEAR] == 2016].index.to_numpy()
    df.loc[idx1, "lead_time"] = df.loc[idx1, "lead_time"] + 10
    gt["error1_leadtime_2016"] = sorted(int(i) for i in idx1)
    # E2: deposit_type -> "Non Refund" for TA/TO channel in 2017     [categorical shift]
    idx2 = df[(df[YEAR] == 2017) & (df["distribution_channel"] == "TA/TO")].index.to_numpy()
    df.loc[idx2, "deposit_type"] = "Non Refund"
    gt["error2_deposit_tato_2017"] = sorted(int(i) for i in idx2)
    # E3: 70% of PRT rows outside 2015 -> country NaN                [NaN corruption]
    cand = df[(df["country"] == "PRT") & (df[YEAR] != 2015)].index.to_numpy()
    idx3 = rng.choice(cand, size=int(len(cand) * 0.70), replace=False)
    df.loc[idx3, "country"] = np.nan
    gt["error3_country_nan"] = sorted(int(i) for i in idx3)
    return df, gt


def analyze(cleaned, clean_t, dirty_t, gt, o2p):
    def eq(a, b):
        if pd.isna(a) and pd.isna(b): return True
        try: return abs(float(a) - float(b)) < 1e-6
        except (TypeError, ValueError): return str(a) == str(b)
    out = []
    e1 = [o2p[i] for i in gt["error1_leadtime_2016"] if i in o2p]
    ch1 = sum(1 for p in e1 if not eq(cleaned.loc[p, "lead_time"], dirty_t.loc[p, "lead_time"]))
    ex1 = sum(1 for p in e1 if eq(cleaned.loc[p, "lead_time"], clean_t.loc[p, "lead_time"]))
    ap1 = sum(1 for p in e1 if abs(float(cleaned.loc[p, "lead_time"]) - float(clean_t.loc[p, "lead_time"])) <= 1.0)
    out.append(f"ERROR 1 (lead_time +10, 2016), {len(e1)} rows: changed {ch1}/{len(e1)} | exactly restored {ex1}/{len(e1)} | within +-1 {ap1}/{len(e1)}")
    e2 = [o2p[i] for i in gt["error2_deposit_tato_2017"] if i in o2p]
    e2c = [p for p in e2 if str(clean_t.loc[p, "deposit_type"]) != "Non Refund"]   # rows whose value really changed
    ch2 = sum(1 for p in e2c if str(cleaned.loc[p, "deposit_type"]) != "Non Refund")
    ex2 = sum(1 for p in e2c if eq(cleaned.loc[p, "deposit_type"], clean_t.loc[p, "deposit_type"]))
    out.append(f"ERROR 2 (deposit->Non Refund TA/TO 2017), {len(e2c)} truly-changed rows: no longer Non Refund {ch2}/{len(e2c)} | exact original category {ex2}/{len(e2c)}")
    e3 = [o2p[i] for i in gt["error3_country_nan"] if i in o2p]
    ch3 = sum(1 for p in e3 if pd.notna(cleaned.loc[p, "country"]))
    ex3 = sum(1 for p in e3 if str(cleaned.loc[p, "country"]) == "PRT")
    out.append(f"ERROR 3 (country NaN, 70% PRT), {len(e3)} rows: no longer missing {ch3}/{len(e3)} | restored to PRT {ex3}/{len(e3)}")
    return out


CONFIG = DatasetConfig(
    name="hotel",
    raw_path="data/hotel_bookings.csv",
    target="is_canceled",
    description=("Hotel booking demand dataset. Each row is a booking. Columns: hotel (City/Resort), "
                 "is_canceled (target), lead_time (days between booking and arrival), arrival_date_year, "
                 "arrival_date_month, arrival_date_week_number, arrival_date_day_of_month, "
                 "stays_in_weekend_nights, stays_in_week_nights, adults, children, babies, meal, "
                 "country (ISO code of guest origin, e.g. PRT, GBR, FRA), market_segment, "
                 "distribution_channel (e.g. TA/TO = travel agents/tour operators, Direct, Corporate), "
                 "is_repeated_guest, previous_cancellations, previous_bookings_not_canceled, "
                 "reserved_room_type, assigned_room_type, booking_changes, deposit_type "
                 "(No Deposit / Non Refund / Refundable), agent, company, days_in_waiting_list, "
                 "customer_type, adr (average daily rate), required_car_parking_spaces, "
                 "total_of_special_requests, reservation_status, reservation_status_date."),
    hints={
        "none": "",
        "weak": ("\n\nHint: Errors are in the lead time, deposit and country columns, there are "
                 "no errors in any entries from 2015."),
        "strong": ("\n\nHint: Errors are here: There is a systematic bias in the lead time of 2016, "
                   "the deposit with distribution channel TA/TO looks wrong in 2017 and often when "
                   "people arrive from PRT, the country is not recorded."),
    },
    numeric_features=["lead_time", "arrival_date_year", "stays_in_week_nights", "adults", "adr",
                      "booking_changes", "total_of_special_requests"],
    categorical_features=["deposit_type", "customer_type", "market_segment", "country"],
    id_cols=["agent", "company", "reservation_status_date"],
    top_k={"country": 15, "market_segment": 8},
    load=load, inject=inject, analyze=analyze,
)
