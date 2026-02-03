"""
Simulation de scénarios — épidémie, grève, canicule, afflux massif.
Applique les paramètres des scénarios (config) sur les prévisions ou séries historiques.
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Optional, Dict, Any
from pathlib import Path

try:
    from config.constants import SCENARIOS, CAPACITE_LITS_TOTALE, SEUIL_ALERTE_OCCUPATION
except ImportError:
    SCENARIOS = {
        "epidemie_grippe": {"surplus_admissions": 0.35, "duree_jours": 45, "pic_jour": 15},
        "greve": {"reduction_personnel": 0.25, "reduction_capacite_lits": 0.15, "duree_jours": 14},
        "canicule": {"surplus_admissions": 0.15, "duree_jours": 21},
        "afflux_massif": {"surplus_admissions": 0.80, "duree_jours": 3, "pic_jour": 1},
    }
    CAPACITE_LITS_TOTALE = 1800
    SEUIL_ALERTE_OCCUPATION = 0.85


def _curve_pic(jours: int, pic_jour: int, duree: int) -> np.ndarray:
    """Courbe en cloche centrée sur pic_jour (indice 0..duree-1)."""
    x = np.arange(duree)
    sigma = max(1, duree / 4)
    return np.exp(-0.5 * ((x - pic_jour) / sigma) ** 2)


def apply_scenario_admissions(
    base_series: pd.Series,
    scenario_key: str,
    start_date: Optional[pd.Timestamp] = None,
    duree_jours_override: Optional[int] = None,
) -> pd.DataFrame:
    """
    Applique un scénario de surplus d'admissions sur une série de base.
    base_series : série quotidienne (admissions).
    duree_jours_override : si fourni, remplace la durée par défaut du scénario (pour des courbes plus longues).
    Retourne un DataFrame avec date, base, scenario, surplus.
    """
    conf = SCENARIOS.get(scenario_key, {})
    if not conf:
        return pd.DataFrame()

    duree = duree_jours_override if duree_jours_override is not None else conf.get("duree_jours", 14)
    duree = max(7, min(duree, 365))  # bornes 7–365 jours
    surplus = conf.get("surplus_admissions", 0.2)
    pic_jour = conf.get("pic_jour", duree // 2)

    start = start_date or base_series.index[-1] + timedelta(days=1)
    dates = pd.date_range(start, periods=duree, freq="D")
    base_mean = base_series.iloc[-28:].mean() if len(base_series) >= 28 else base_series.mean()
    curve = _curve_pic(0, pic_jour, duree)
    curve = curve / curve.max() if curve.max() > 0 else curve

    rows = []
    for i, d in enumerate(dates):
        mult = 1 + surplus * curve[i]
        val = base_mean * mult
        rows.append({
            "date": d,
            "admissions_base": base_mean,
            "admissions_scenario": val,
            "surplus_pct": (mult - 1) * 100,
        })
    return pd.DataFrame(rows)


def apply_scenario_greve(
    occupation_quotidienne: pd.DataFrame,
    scenario_key: str = "greve",
    start_date: Optional[pd.Timestamp] = None,
    duree_jours_override: Optional[int] = None,
) -> pd.DataFrame:
    """
    Scénario grève : réduction progressive de la capacité et accumulation du flux.
    duree_jours_override : si fourni, remplace la durée par défaut (pour des courbes plus longues).
    """
    conf = SCENARIOS.get(scenario_key, {})
    if not conf:
        return pd.DataFrame()

    reduction_lits = conf.get("reduction_capacite_lits", 0.15)
    duree = duree_jours_override if duree_jours_override is not None else conf.get("duree_jours", 14)
    duree = max(7, min(duree, 365))
    start = start_date or occupation_quotidienne["date"].max() + timedelta(days=1)
    dates = pd.date_range(start, periods=duree, freq="D")

    occ_mean = occupation_quotidienne["occupation_lits"].iloc[-28:].mean()
    # Accumulation quotidienne (retards de sortie, moindre rotation) : +0,5 % à 1,5 % par jour
    croissance_quotidienne = 0.008  # 0,8 % par jour

    rows = []
    for i, d in enumerate(dates):
        # Ramp-up de l'effet grève sur 2–3 jours, puis plateau, légère baisse en fin
        if i < 2:
            facteur_reduction = reduction_lits * (i + 1) / 2
        elif i < duree - 2:
            facteur_reduction = reduction_lits
        else:
            facteur_reduction = reduction_lits * 0.85  # légère reprise en fin
        capacite_effective = CAPACITE_LITS_TOTALE * (1 - facteur_reduction)
        # Occupation qui augmente légèrement (effet file d'attente)
        occupation_jour = occ_mean * (1 + croissance_quotidienne * i)
        taux_eff = occupation_jour / capacite_effective if capacite_effective > 0 else 0
        rows.append({
            "date": d,
            "occupation_lits": occupation_jour,
            "capacite_effective": capacite_effective,
            "taux_occupation_effectif": taux_eff,
            "scenario": scenario_key,
        })
    return pd.DataFrame(rows)


def run_scenario(
    scenario_key: str,
    occupation_df: pd.DataFrame,
    admissions_df: Optional[pd.DataFrame] = None,
    start_date: Optional[pd.Timestamp] = None,
    duree_jours_override: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Lance un scénario et retourne prévisions + indicateurs.
    duree_jours_override : durée en jours (7–365). Si fourni, remplace la durée par défaut du scénario.
    """
    if scenario_key == "greve":
        result_df = apply_scenario_greve(
            occupation_df, scenario_key=scenario_key, start_date=start_date, duree_jours_override=duree_jours_override
        )
        taux_max = result_df["taux_occupation_effectif"].max()
    else:
        if admissions_df is not None:
            adm_series = admissions_df.groupby("date")["admissions"].sum()
        else:
            adm_series = occupation_df.set_index("date")["admissions_jour"]
        result_df = apply_scenario_admissions(
            adm_series, scenario_key, start_date=start_date, duree_jours_override=duree_jours_override
        )
        if not result_df.empty:
            result_df["occupation_estimee"] = (
                result_df["admissions_scenario"] * 6
            )  # approx 6 j séjour
            result_df["taux_occupation_estime"] = (
                result_df["occupation_estimee"] / CAPACITE_LITS_TOTALE
            )
            taux_max = result_df["taux_occupation_estime"].max()
        else:
            taux_max = 0

    alerte = "critique" if taux_max >= 0.95 else ("alerte" if taux_max >= SEUIL_ALERTE_OCCUPATION else "normal")
    return {
        "scenario": scenario_key,
        "label": SCENARIOS.get(scenario_key, {}).get("label", scenario_key),
        "resultat": result_df,
        "taux_occupation_max": float(taux_max),
        "alerte": alerte,
    }
