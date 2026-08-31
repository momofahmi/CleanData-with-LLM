from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class DatasetConfig:
    name: str
    raw_path: str                       # where the user drops the raw CSV
    target: str                         # classification target for the fixed evaluator
    description: str                    # NEUTRAL column description (no hint leakage)
    hints: Dict[str, str]               # none / weak / strong (paper wording)
    numeric_features: List[str]
    categorical_features: List[str]
    id_cols: List[str] = field(default_factory=list)   # excluded from stats + eval
    entity_col: Optional[str] = None    # panel data (e.g. country)
    time_col: Optional[str] = None      # panel data (e.g. year)
    load: Optional[Callable] = None     # (path)->clean df, canonicalised
    inject: Optional[Callable] = None   # (clean_df, seed)->(dirty_df, ground_truth dict)
    analyze: Optional[Callable] = None  # (cleaned, clean_train, dirty_train, gt)->list[str]
    top_k: Dict[str, int] = field(default_factory=dict)  # one-hot caps per categorical
