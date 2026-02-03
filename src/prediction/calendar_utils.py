"""
Calendrier — Jours fériés et vacances scolaires (réf. Bouteloup 2020).
Zone C (Paris / Île-de-France) pour Pitié-Salpêtrière.
Utilisé comme variables dans la régression pour améliorer la prédiction du flux.
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Optional

# Jours fériés fixes (France)
_FERIES_FIXES = [(1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)]


def _easter_year(y: int) -> date:
    """Date de Pâques (algorithme de Meeus)."""
    a = y % 19
    b = y // 100
    c = y % 100
    d = b // 4
    e = b % 4
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    j = c // 4
    k = c % 4
    m = (a + 11 * h) // 319
    r = (2 * e + 2 * j - k - h + m + 32) % 7
    n = (h - m + r + 90) // 25
    p = (h - m + r + n + 19) % 32
    return date(y, n, p)


def is_jour_ferie(d: pd.Timestamp) -> bool:
    """True si la date est un jour férié français."""
    if isinstance(d, pd.Timestamp):
        d = d.to_pydatetime().date()
    month, day = d.month, d.day
    if (month, day) in _FERIES_FIXES:
        return True
    year = d.year
    easter = _easter_year(year)
    # Lundi Pâques, Ascension (E+39), Lundi Pentecôte (E+50)
    for delta in [1, 39, 50]:
        if d == easter + timedelta(days=delta):
            return True
    return False


def is_veille_ferie(d: pd.Timestamp) -> bool:
    """True si le lendemain est un jour férié (réf. Bouteloup)."""
    if isinstance(d, pd.Timestamp):
        next_d = d + pd.Timedelta(days=1)
    else:
        next_d = d + timedelta(days=1)
    return is_jour_ferie(next_d)


def is_lendemain_ferie(d: pd.Timestamp) -> bool:
    """True si la veille est un jour férié (réf. Bouteloup)."""
    if isinstance(d, pd.Timestamp):
        prev_d = d - pd.Timedelta(days=1)
    else:
        prev_d = d - timedelta(days=1)
    return is_jour_ferie(prev_d)


# Vacances scolaires zone C (Paris) — périodes approximatives (début, fin) par année
# Format: (mois_debut, jour_debut, mois_fin, jour_fin)
_VACANCES_ZONE_C = [
    (10, 22, 11, 6),   # Toussaint
    (12, 23, 1, 7),    # Noël
    (2, 11, 2, 27),    # Hiver
    (4, 8, 4, 24),     # Printemps
    (7, 8, 9, 4),      # Été
]


def _in_period(d: date, mo1: int, j1: int, mo2: int, j2: int, year: int) -> bool:
    """True si d est dans la période (mo1-j1) à (mo2-j2) pour l'année (gère le passage année)."""
    try:
        start = date(year, mo1, j1)
        end = date(year + 1, mo2, j2) if mo1 > mo2 else date(year, mo2, j2)
    except ValueError:
        return False
    if start <= end:
        return start <= d <= end
    # passage d'année (ex: 23 déc - 7 janv)
    return d >= start or d <= end


def is_vacances_scolaires_zone_c(d: pd.Timestamp) -> bool:
    """True si la date est en vacances scolaires zone C (Paris). Réf. Bouteloup."""
    if isinstance(d, pd.Timestamp):
        d = d.to_pydatetime().date()
    y = d.year
    for mo1, j1, mo2, j2 in _VACANCES_ZONE_C:
        if _in_period(d, mo1, j1, mo2, j2, y):
            return True
        if mo1 > mo2 and _in_period(d, mo1, j1, mo2, j2, y + 1):
            return True
    return False


def temperature_synthetique_paris(d: pd.Timestamp) -> float:
    """
    Température quotidienne approximative (Paris) pour structure météo (réf. Bouteloup).
    Déterministe : courbe sinusoïdale (min ~5°C janv, max ~25°C juill). En °C.
    """
    import math
    if isinstance(d, pd.Timestamp):
        day_of_year = d.timetuple().tm_yday
    else:
        day_of_year = (d.month - 1) * 31 + d.day
    rad = 2 * math.pi * (day_of_year - 15) / 365
    return 15.0 + 10.0 * math.sin(rad)


def add_calendar_features(df: pd.DataFrame, date_index: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
    """
    Ajoute les colonnes calendrier (réf. Bouteloup) à un DataFrame dont l'index est date.
    Si date_index est None, utilise df.index. Modifie df en place et retourne df.
    """
    idx = date_index if date_index is not None else df.index
    df["jour_ferie"] = np.array([1 if is_jour_ferie(d) else 0 for d in idx])
    df["veille_ferie"] = np.array([1 if is_veille_ferie(d) else 0 for d in idx])
    df["lendemain_ferie"] = np.array([1 if is_lendemain_ferie(d) else 0 for d in idx])
    df["vacances_scolaires"] = np.array([1 if is_vacances_scolaires_zone_c(d) else 0 for d in idx])
    return df
