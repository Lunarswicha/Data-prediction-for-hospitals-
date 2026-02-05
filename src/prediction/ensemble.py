"""
Système d'ensemble professionnel pour prévisions hospitalières.
Agrégation pondérée des meilleurs modèles pour maximiser la robustesse.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple
from .models import (
    predict_holt_winters,
    predict_regression,
    predict_sarima,
    predict_boosting,
    run_backtest_admissions,
)


def predict_admissions_ensemble(
    series: pd.Series,
    horizon_jours: int = 14,
) -> pd.DataFrame:
    """
    Prévision par ensemble pondéré des 3 meilleurs modèles.
    
    Méthode:
    1. Backtest sur 90 jours pour évaluer chaque modèle
    2. Sélection des 3 meilleurs (% within ±10%)
    3. Pondération par performance inverse (meilleur = poids plus élevé)
    4. Agrégation pondérée des prédictions
    
    Returns:
        DataFrame avec colonnes: date, prediction, prediction_low, prediction_high
    """
    # Tous les modèles disponibles
    modeles = {
        "holt_winters": predict_holt_winters,
        "regression": predict_regression,
        "sarima": predict_sarima,
        "boosting": predict_boosting,
    }
    
    # 1. Backtest pour évaluer chaque modèle
    validation_days = min(90, max(28, len(series) // 3))
    performances = {}
    
    for nom, fonction in modeles.items():
        try:
            metrics = run_backtest_admissions(series, fonction, validation_days=validation_days)
            if metrics and "pct_within_10" in metrics:
                performances[nom] = {
                    "fonction": fonction,
                    "pct_within_10": metrics["pct_within_10"],
                    "mae": metrics.get("mae", 999),
                }
        except Exception:
            continue
    
    if len(performances) == 0:
        # Fallback : Holt-Winters seul
        return predict_holt_winters(series, horizon_jours=horizon_jours)
    
    # 2. Sélectionner top 3 par % within ±10%
    top3 = sorted(performances.items(), key=lambda x: x[1]["pct_within_10"], reverse=True)[:3]
    
    if len(top3) == 0:
        return predict_holt_winters(series, horizon_jours=horizon_jours)
    
    # 3. Calculer pondérations (proportionnel au score)
    total_score = sum([p[1]["pct_within_10"] for p in top3])
    if total_score == 0:
        weights = {p[0]: 1.0 / len(top3) for p in top3}
    else:
        weights = {p[0]: p[1]["pct_within_10"] / total_score for p in top3}
    
    # 4. Générer prédictions de chaque modèle
    predictions = {}
    for nom, _ in top3:
        try:
            pred = performances[nom]["fonction"](series, horizon_jours=horizon_jours)
            if pred is not None and len(pred) >= horizon_jours:
                predictions[nom] = pred["prediction"].values[:horizon_jours]
        except Exception:
            continue
    
    if len(predictions) == 0:
        return predict_holt_winters(series, horizon_jours=horizon_jours)
    
    # 5. Agrégation pondérée
    ensemble_pred = np.zeros(horizon_jours)
    for nom, pred_values in predictions.items():
        ensemble_pred += weights.get(nom, 0) * pred_values
    
    # 6. Intervalles de confiance : écart-type entre modèles
    all_preds = np.array(list(predictions.values()))
    std_between_models = np.std(all_preds, axis=0)
    
    # IC95% = moyenne ± 1.96 * std
    ic_low = ensemble_pred - 1.96 * std_between_models
    ic_high = ensemble_pred + 1.96 * std_between_models
    
    # Date range
    dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon_jours, freq="D")
    
    return pd.DataFrame({
        "date": dates,
        "prediction": ensemble_pred,
        "prediction_low": np.maximum(0, ic_low),
        "prediction_high": ic_high,
        "type": "admissions",
    })


def get_ensemble_info(series: pd.Series) -> dict:
    """
    Retourne les informations sur les modèles de l'ensemble.
    Pour affichage dans le dashboard.
    """
    modeles = {
        "Holt-Winters": predict_holt_winters,
        "Régression Ridge": predict_regression,
        "SARIMA": predict_sarima,
        "Boosting XGBoost": predict_boosting,
    }
    
    validation_days = min(90, max(28, len(series) // 3))
    performances = {}
    
    for nom, fonction in modeles.items():
        try:
            metrics = run_backtest_admissions(series, fonction, validation_days=validation_days)
            if metrics:
                performances[nom] = {
                    "pct_within_10": metrics.get("pct_within_10", 0) * 100,
                    "mae": metrics.get("mae", 0),
                    "rmse": metrics.get("rmse", 0),
                }
        except Exception:
            performances[nom] = {
                "pct_within_10": 0,
                "mae": 999,
                "rmse": 999,
            }
    
    # Top 3
    top3 = sorted(performances.items(), key=lambda x: x[1]["pct_within_10"], reverse=True)[:3]
    
    # Pondérations
    total_score = sum([p[1]["pct_within_10"] for p in top3])
    if total_score > 0:
        weights = {p[0]: p[1]["pct_within_10"] / total_score * 100 for p in top3}
    else:
        weights = {p[0]: 100.0 / len(top3) for p in top3}
    
    return {
        "top3_names": [p[0] for p in top3],
        "weights": weights,
        "all_performances": performances,
        "validation_days": validation_days,
    }
