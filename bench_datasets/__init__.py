from .base import DatasetConfig
from . import titanic, hotel, meat, adult

REGISTRY = {"titanic": titanic.CONFIG, "hotel": hotel.CONFIG, "meat": meat.CONFIG, "adult": adult.CONFIG}

def get(name):
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; choose from {list(REGISTRY)}")
    return REGISTRY[name]
