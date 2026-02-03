"""Chargement des données (générées ou processed)."""

import pandas as pd
from pathlib import Path
from typing import Optional


def get_data_dir(base: Optional[Path] = None) -> Path:
    if base is None:
        base = Path(__file__).resolve().parents[2]
    return base / "data"


def load_admissions(data_dir: Optional[Path] = None) -> pd.DataFrame:
    path = (data_dir or get_data_dir()) / "generated" / "admissions_quotidiennes_par_service.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier non trouvé: {path}. Exécuter d'abord: python -m src.data.generate"
        )
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_occupation(data_dir: Optional[Path] = None) -> pd.DataFrame:
    path = (data_dir or get_data_dir()) / "generated" / "occupation_quotidienne.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier non trouvé: {path}. Exécuter d'abord: python -m src.data.generate"
        )
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_all(data_dir: Optional[Path] = None) -> tuple:
    return load_admissions(data_dir), load_occupation(data_dir)
