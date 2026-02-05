"""
Génération du jeu de données fictif — Pitié-Salpêtrière.
Tendances réalistes : saisonnalité (grippe hiver, etc.), jour de la semaine, services.
Données 100 % synthétiques, aucune donnée réelle.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Répartition des admissions par service (ordres de grandeur fictifs)
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

# Saisonnalité mensuelle (indice 1 = moyenne) — hiver plus chargé
MONTHLY_INDEX = {
    1: 1.18, 2: 1.12, 3: 1.05, 4: 0.98, 5: 0.95, 6: 0.92,
    7: 0.90, 8: 0.92, 9: 0.98, 10: 1.02, 11: 1.08, 12: 1.15,
}

# Lundi = 0, Dimanche = 6 — semaine plus chargée en début de semaine
WEEKDAY_INDEX = [1.05, 1.08, 1.04, 1.02, 1.00, 0.88, 0.82]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def generate_admissions(
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",  # Au moins 2 années complètes pour comparer hiver / été
    daily_base: int = 320,
    trend_per_year: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Génère des admissions quotidiennes par service avec saisonnalité et bruit.
    CORRECTION 5 FÉV 2026 : Augmentation du bruit et ajout d'événements aléatoires
    pour éviter que tous les modèles de ML convergent vers les mêmes prédictions.
    """
    np.random.seed(seed)
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    dates = pd.date_range(start, end, freq="D")

    n_days = len(dates)
    # Pas de tendance : niveau stable, seule la saisonnalité (mois, jour de la semaine) varie
    trend = np.linspace(0, trend_per_year * (n_days / 365), n_days) if trend_per_year != 0 else np.zeros(n_days)
    
    # Saisonnalité mensuelle : hiver > été, MAIS avec variation aléatoire ±15%
    month_idx = np.array([MONTHLY_INDEX[d.month] * np.random.uniform(0.85, 1.15) for d in dates])
    
    # Saisonnalité hebdomadaire : base + bruit pour éviter rigidité
    weekday_idx = np.array([WEEKDAY_INDEX[d.weekday()] * np.random.uniform(0.90, 1.10) for d in dates])
    
    # Bruit augmenté : 18% au lieu de 8% (données réelles hospitalières plus volatiles)
    noise = np.random.normal(1, 0.18, n_days)
    
    # Composante AR(1) pour autocorrélation (les admissions d'un jour dépendent du jour précédent)
    ar_component = np.zeros(n_days)
    ar_component[0] = np.random.normal(0, 0.1)
    for i in range(1, n_days):
        ar_component[i] = 0.3 * ar_component[i-1] + np.random.normal(0, 0.1)
    
    # Événements aléatoires (pics imprévisibles : épidémie, accident collectif, etc.)
    # 5% des jours ont un événement (+20 à +80 admissions)
    random_events = np.zeros(n_days)
    n_events = int(n_days * 0.05)
    event_days = np.random.choice(n_days, n_events, replace=False)
    for day in event_days:
        random_events[day] = np.random.uniform(20, 80)
    
    daily_total = (
        daily_base
        * (1 + trend)
        * month_idx
        * weekday_idx
        * np.clip(noise, 0.5, 1.5)  # Plage élargie (50%-150% au lieu de 70%-130%)
        * (1 + ar_component)
        + random_events
    ).astype(int)

    rows = []
    for i, d in enumerate(dates):
        n = max(10, daily_total[i])
        # Répartition par service avec léger bruit
        shares = np.array(list(SERVICE_SHARE.values()))
        noise_s = np.random.dirichlet(np.ones(8) * 10)
        shares = shares * noise_s / noise_s.sum() * (shares.sum() / shares.sum())
        counts = (np.random.multinomial(n, shares / shares.sum())).tolist()
        for service, count in zip(SERVICE_SHARE.keys(), counts):
            rows.append({"date": d, "service": service, "admissions": count})

    return pd.DataFrame(rows)


def generate_occupation(
    admissions_df: pd.DataFrame,
    duree_sejour_moyenne_jours: float = 6.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simule un taux d'occupation des lits à partir des admissions (modèle simplifié).
    """
    np.random.seed(seed)
    # Lits par service (ordre aligné)
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
    # Occupation approximative = stock cumulé avec sorties (modèle exponentiel)
    alpha = 1 / duree_sejour_moyenne_jours
    occupation = []
    stock = {s: 0.0 for s in lits_service}
    for _, row in daily.iterrows():
        day_adm = admissions_df[admissions_df["date"] == row["date"]]
        for _, r in day_adm.iterrows():
            stock[r["service"]] = stock.get(r["service"], 0) + r["admissions"]
        for s in lits_service:
            stock[s] = stock[s] * (1 - alpha) + np.random.uniform(0, 2)
            stock[s] = max(0, min(lits_service[s], stock[s]))
        occ_total = sum(stock.values())
        occupation.append(
            {
                "date": row["date"],
                "occupation_lits": occ_total,
                "taux_occupation": occ_total / total_lits,
                "admissions_jour": row["admissions"],
            }
        )

    return pd.DataFrame(occupation)


def generate_all(
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    output_dir: Optional[Path] = None,
) -> tuple:
    """
    Génère les données fictives et les enregistre (optionnel).
    Par défaut : 3 ans (2022–2024) pour une vue annuelle et la comparaison hiver / été.
    Retourne (admissions par date/service, occupation quotidienne).
    """
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
