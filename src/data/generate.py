"""
Génération du jeu de données fictif — Pitié-Salpêtrière.
Données 100 % synthétiques, inspirées des ordres de grandeur et patterns
de la Pitié-Salpêtrière : saisonnalité marquée (hiver > été), virus, canicule,
grève, durée de séjour réaliste, passages urgences.
Structure des sorties inchangée pour compatibilité dashboard et modèles.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Répartition des admissions par service (ordres de grandeur type CHU Paris)
SERVICE_SHARE = {
    "urgences": 0.28,
    "cardiologie": 0.12,
    "neurologie": 0.10,
    "maladies_infectieuses": 0.06,
    "reanimation": 0.04,
    "pediatrie": 0.12,
    "chirurgie": 0.14,
    "medecine_generale": 0.14,
}

# Saisonnalité mensuelle forte : hiver (déc–mars) > été (juin–août) — réaliste hôpital
# Indice 1 = moyenne annuelle ; hiver grippe/bronchiolite, été baisse activité
MONTHLY_INDEX = {
    1: 1.22, 2: 1.18, 3: 1.12, 4: 1.02, 5: 0.96, 6: 0.88,
    7: 0.84, 8: 0.86, 9: 0.94, 10: 1.02, 11: 1.12, 12: 1.18,
}

# Lundi = 0 … Dimanche = 6 — semaine plus chargée en début de semaine, week-end plus calme
WEEKDAY_INDEX = [1.08, 1.06, 1.04, 1.02, 1.00, 0.90, 0.82]

# DMS (durée moyenne de séjour) par mois en jours — plus long en hiver (pathologies lourdes)
DMS_BY_MONTH = {
    1: 6.8, 2: 6.6, 3: 6.4, 4: 6.0, 5: 5.8, 6: 5.5,
    7: 5.4, 8: 5.5, 9: 5.8, 10: 6.0, 11: 6.4, 12: 6.6,
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _flu_boost(month: int) -> float:
    """Surcroît grippe / bronchiolite (nov–mars)."""
    if month in (11, 12, 1, 2, 3):
        return np.random.uniform(1.02, 1.12)
    return 1.0


def _canicule_effect(month: int, day: int) -> float:
    """Canicule été : baisse activité programmée, pic déshydratation possible."""
    if month not in (6, 7, 8):
        return 1.0
    # Quelques jours de canicule par été : léger surcroît urgences puis baisse
    if np.random.random() < 0.08:
        return np.random.uniform(0.92, 1.05)
    return 1.0


def _strike_effect(dates: pd.DatetimeIndex, seed: int) -> np.ndarray:
    """Jours de grève : baisse des admissions (blocage partiel)."""
    np.random.seed(seed)
    n_days = len(dates)
    out = np.ones(n_days)
    # 2 à 4 épisodes de 3–7 jours sur la période
    n_episodes = np.random.randint(2, 5)
    for _ in range(n_episodes):
        start = np.random.randint(0, max(1, n_days - 10))
        length = np.random.randint(3, 8)
        for j in range(start, min(start + length, n_days)):
            out[j] *= np.random.uniform(0.72, 0.88)
    return out


def _virus_wave(dates: pd.DatetimeIndex, seed: int) -> np.ndarray:
    """Vague type épidémie : pic localisé sur quelques semaines."""
    np.random.seed(seed + 1)
    n_days = len(dates)
    out = np.ones(n_days)
    # Une vague par an environ, plutôt en hiver/début printemps
    for year in pd.Series(dates).dt.year.unique():
        idx_year = np.where(pd.Series(dates).dt.year == year)[0]
        if len(idx_year) < 60:
            continue
        # Début de vague aléatoire entre nov et mars
        start_in_year = np.random.randint(0, min(120, len(idx_year) - 40))
        start = idx_year[start_in_year]
        length = np.random.randint(14, 35)
        for j in range(start, min(start + length, n_days)):
            # Pic en milieu de vague
            progress = (j - start) / max(1, length)
            mult = 1.0 + 0.15 * np.exp(-((progress - 0.5) ** 2) / 0.08)
            out[j] *= np.clip(mult * np.random.uniform(0.95, 1.05), 0.9, 1.25)
    return out


def generate_admissions(
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    daily_base: int = 320,
    trend_per_year: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Génère des admissions quotidiennes par service.
    Saisonnalité marquée hiver/été, jour de la semaine, grippe, canicule, grève, vague épidémie.
    Structure inchangée : date, service, admissions.
    """
    np.random.seed(seed)
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    dates = pd.date_range(start, end, freq="D")
    n_days = len(dates)

    trend = np.linspace(0, trend_per_year * (n_days / 365), n_days) if trend_per_year != 0 else np.zeros(n_days)
    month_idx = np.array([MONTHLY_INDEX[d.month] for d in dates])
    weekday_idx = np.array([WEEKDAY_INDEX[d.weekday()] for d in dates])
    flu = np.array([_flu_boost(d.month) for d in dates])
    canicule = np.array([_canicule_effect(d.month, d.day) for d in dates])
    strike = _strike_effect(dates, seed)
    virus = _virus_wave(dates, seed)
    noise = np.random.normal(1, 0.12, n_days)
    ar = np.zeros(n_days)
    ar[0] = np.random.normal(0, 0.08)
    for i in range(1, n_days):
        ar[i] = 0.25 * ar[i - 1] + np.random.normal(0, 0.08)

    daily_total = (
        daily_base
        * (1 + trend)
        * month_idx
        * weekday_idx
        * flu
        * canicule
        * strike
        * virus
        * np.clip(noise, 0.6, 1.4)
        * (1 + ar)
    ).astype(int)

    rows = []
    for i, d in enumerate(dates):
        n = max(15, daily_total[i])
        shares = np.array(list(SERVICE_SHARE.values()))
        noise_s = np.random.dirichlet(np.ones(8) * 12)
        shares = shares * noise_s / (noise_s.sum() * (shares.sum() / shares.sum()))
        counts = np.random.multinomial(n, shares / shares.sum()).tolist()
        for service, count in zip(SERVICE_SHARE.keys(), counts):
            rows.append({"date": d, "service": service, "admissions": max(0, count)})

    return pd.DataFrame(rows)


def generate_occupation(
    admissions_df: pd.DataFrame,
    duree_sejour_moyenne_jours: float = 6.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simule l'occupation des lits à partir des admissions.
    DMS saisonnière (plus long en hiver), même structure : date, occupation_lits, taux_occupation, admissions_jour.
    """
    np.random.seed(seed)
    lits_service = {
        "urgences": 140,
        "cardiologie": 220,
        "neurologie": 180,
        "maladies_infectieuses": 140,
        "reanimation": 110,
        "pediatrie": 180,
        "chirurgie": 400,
        "medecine_generale": 430,
    }
    total_lits = sum(lits_service.values())

    daily = (
        admissions_df.groupby("date")
        .agg(admissions=("admissions", "sum"))
        .reset_index()
    )
    occupation = []
    stock = {s: 0.0 for s in lits_service}
    for _, row in daily.iterrows():
        d = row["date"]
        month = d.month
        dms = DMS_BY_MONTH.get(month, duree_sejour_moyenne_jours)
        alpha = 1.0 / max(2.0, dms)
        day_adm = admissions_df[admissions_df["date"] == row["date"]]
        for _, r in day_adm.iterrows():
            stock[r["service"]] = stock.get(r["service"], 0) + r["admissions"]
        for s in lits_service:
            decay = 1 - alpha * np.random.uniform(0.95, 1.05)
            stock[s] = max(0, min(lits_service[s], stock[s] * decay + np.random.uniform(-1, 2)))
        occ_total = sum(stock.values())
        occupation.append({
            "date": row["date"],
            "occupation_lits": round(occ_total, 1),
            "taux_occupation": round(occ_total / total_lits, 4),
            "admissions_jour": row["admissions"],
        })

    return pd.DataFrame(occupation)


def generate_all(
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    output_dir: Optional[Path] = None,
) -> tuple:
    """Génère admissions et occupation, enregistre en CSV si output_dir fourni. Structure inchangée."""
    admissions = generate_admissions(start_date=start_date, end_date=end_date)
    occupation = generate_occupation(admissions)

    if output_dir is not None:
        _ensure_dir(output_dir)
        admissions.to_csv(output_dir / "admissions_quotidiennes_par_service.csv", index=False)
        occupation.to_csv(output_dir / "occupation_quotidienne.csv", index=False)

    return admissions, occupation


if __name__ == "__main__":
    _project_root = Path(__file__).resolve().parents[2]
    _output_dir = _project_root / "data" / "generated"
    admissions_df, occupation_df = generate_all(output_dir=_output_dir)
    print("Admissions:", admissions_df.shape)
    print(admissions_df.head(10))
    print("\nOccupation:", occupation_df.shape)
    print(occupation_df.tail())
