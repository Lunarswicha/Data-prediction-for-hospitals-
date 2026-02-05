"""
Modèles de prédiction — admissions, occupation, besoins (lits, personnel).
- Baseline : moyenne glissante + tendance.
- Holt-Winters (lissage exponentiel saisonnier) : saisonnalité hebdo + tendance.
- Régression : lags (1, 7, 14 + mean 7–13 réf. Bouteloup) + calendrier (jour_semaine, mois,
  jours fériés, vacances scolaires, température synthétique réf. thèses).
- SARIMA (optionnel) : pour séries longues avec saisonnalité.
- Durée de séjour saisonnière (réf. Lequertier) dans le modèle stock.
Intervalles de confiance ; métrique de validation ±10 % (réf. Bouteloup).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

try:
    from src.prediction.calendar_utils import (
        is_jour_ferie,
        is_veille_ferie,
        is_lendemain_ferie,
        is_vacances_scolaires_zone_c,
        temperature_synthetique_paris,
    )
except ImportError:
    def _no_calendar(d):
        return False
    def _temp_const(d):
        return 15.0
    is_jour_ferie = is_veille_ferie = is_lendemain_ferie = is_vacances_scolaires_zone_c = _no_calendar
    temperature_synthetique_paris = _temp_const

# Saisonnalité mensuelle (alignée données synthétiques / réalité : hiver > été)
_MONTHLY_INDEX = {
    1: 1.18, 2: 1.12, 3: 1.05, 4: 0.98, 5: 0.95, 6: 0.92,
    7: 0.90, 8: 0.92, 9: 0.98, 10: 1.02, 11: 1.08, 12: 1.15,
}
_MONTHLY_INDEX_MEAN = sum(_MONTHLY_INDEX.values()) / 12.0


def smooth_forecast_curve(pred_df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Lisse la courbe de prévision (moyenne glissante) pour horizon > 14 j.
    Supprime les « vagues » hebdomadaires artificielles tout en gardant la tendance.
    """
    if pred_df.empty or len(pred_df) < window:
        return pred_df
    out = pred_df.copy()
    for col in ("occupation_lits_pred", "occupation_lits_low", "occupation_lits_high"):
        if col in out.columns:
            out[col] = out[col].rolling(window=min(window, len(out)), min_periods=1, center=True).mean()
    return out


def _smooth_seasonal_factor_day(d: Any) -> float:
    """
    Facteur saisonnier CONTINU (jour de l'année) : pic hiver, creux été.
    Amplitude ±15 % pour que la courbe soit LISIBLE (pas plate).
    Pas de saut au 1er du mois (sinusoïde = transitions douces).
    """
    ts = pd.Timestamp(d) if not isinstance(d, pd.Timestamp) else d
    doy = ts.dayofyear
    import math
    angle = 2 * math.pi * (doy - 15) / 365.0
    return 1.0 + 0.15 * math.cos(angle)


def _apply_monthly_seasonality_to_occupation(pred_df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Applique une saisonnalité LISSÉE (sinusoïde par jour de l'année), pas par mois.
    Hiver > été, mais sans marche d'escalier au 1er de chaque mois.
    """
    if pred_df.empty:
        return pred_df
    out = pred_df.copy()
    for col in ("occupation_lits_pred", "occupation_lits_low", "occupation_lits_high"):
        if col not in out.columns:
            continue
        factors = out[date_col].apply(_smooth_seasonal_factor_day)
        out[col] = (out[col] * factors).values
    return out


# Préparation des séries
def prepare_series(occupation_df: pd.DataFrame, col: str = "admissions_jour") -> pd.Series:
    """Série temporelle quotidienne avec index date."""
    df = occupation_df[["date", col]].drop_duplicates(subset=("date",)).sort_values("date")
    return cast(pd.Series, df.set_index("date")[col])


def _ensure_daily_index(series: pd.Series) -> pd.Series:
    """Réindexe sur une plage de dates quotidienne continue, remplit les manquants par interpolation."""
    if series.empty:
        return series
    dr = pd.date_range(series.index.min(), series.index.max(), freq="D")
    return series.reindex(dr).interpolate(method="linear").ffill().bfill()


# --- Baseline : moyenne glissante ---
def predict_moving_average(
    series: pd.Series,
    horizon_jours: int = 14,
    window: int = 28,
) -> pd.DataFrame:
    """
    Prévision par moyenne glissante + tendance (baseline).
    Retourne : date, prediction, prediction_low, prediction_high (approximatif).
    """
    series = _ensure_daily_index(series)
    if len(series) < 7:
        last = series.mean()
        trend = 0
        std = series.std() or last * 0.1
    else:
        last = series.iloc[-window:].mean()
        # Pas de tendance extrapolée : évite une dérive linéaire irréaliste sur des horizons longs.
        trend = 0
        std = series.iloc[-window:].std() or (last * 0.08)
    dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon_jours, freq="D")
    pred = []
    for i, d in enumerate(dates):
        val = last + trend * (i + 1)
        val = max(0, val)
        pred.append({
            "date": d,
            "prediction": val,
            "prediction_low": max(0, val - 1.96 * std),
            "prediction_high": val + 1.96 * std,
            "type": "admissions",
        })
    return pd.DataFrame(pred)


# --- Holt-Winters (lissage exponentiel saisonnier) ---
def predict_holt_winters(
    series: pd.Series,
    horizon_jours: int = 14,
    seasonal_period: int = 7,
) -> Optional[pd.DataFrame]:
    """
    Prévision par moyenne historique stable + saisonnalité hebdomadaire.
    CORRECTION : Au lieu d'utiliser Holt-Winters qui peut extrapoler une baisse,
    on force un niveau STABLE = moyenne des 90 derniers jours, puis on applique
    uniquement la saisonnalité hebdomadaire (jour de la semaine).
    """
    series = _ensure_daily_index(series)
    if len(series) < seasonal_period * 2:
        return None

    # Niveau stable = moyenne historique récente (90 derniers jours)
    window = min(90, len(series))
    niveau_stable = series.iloc[-window:].mean()
    
    # Extraire la saisonnalité hebdomadaire (moyenne par jour de la semaine)
    series_recent = series.iloc[-min(84, len(series)):]  # ~12 semaines
    saisonnalite_hebdo = {}
    for dow in range(7):  # 0=lundi, 6=dimanche
        days_of_week = series_recent[series_recent.index.dayofweek == dow]
        if len(days_of_week) > 0:
            # Écart à la moyenne globale
            saisonnalite_hebdo[dow] = days_of_week.mean() - niveau_stable
        else:
            saisonnalite_hebdo[dow] = 0.0
    
    # Générer les prédictions : niveau stable + saisonnalité jour de la semaine
    dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon_jours, freq="D")
    predictions = []
    for d in dates:
        dow = d.dayofweek
        pred = niveau_stable + saisonnalite_hebdo.get(dow, 0.0)
        predictions.append(max(0, pred))
    
    # Intervalles de confiance basés sur la variabilité historique
    std_recent = series.iloc[-window:].std()
    predictions_arr = np.array(predictions)
    low = np.maximum(0, predictions_arr - 1.96 * std_recent)
    high = predictions_arr + 1.96 * std_recent

    return pd.DataFrame({
        "date": dates,
        "prediction": predictions_arr,
        "prediction_low": low,
        "prediction_high": high,
        "type": "admissions",
    })


# --- Régression : lags + calendrier (réf. Bouteloup : lags 7–13, jours fériés, vacances, météo) ---
def _build_lag_calendar_features(series: pd.Series, horizon: int, use_calendar_ferie: bool = True, use_temperature: bool = True) -> Tuple:
    """
    Construit les features pour X_train et X_pred.
    Lags 1, 7, 14 + mean(j-7 à j-13) (Bouteloup) + jour_semaine, mois + jours fériés, vacances (Bouteloup)
    + température synthétique (optionnel) + saison_grippe (nov–mars), mois_hiver (déc–fév) pour pics épidémiques.
    Retourne (X_train, y_train, X_pred, pred_dates).
    """
    series = _ensure_daily_index(series)
    df = pd.DataFrame({"y": series})
    for lag in [1, 7, 14]:
        if len(series) > lag:
            df["lag_{}".format(lag)] = df["y"].shift(lag)
    # Moyenne des passages j-7 à j-13 (réf. Bouteloup)
    if len(series) >= 14:
        lag_7_13 = sum(df["y"].shift(k) for k in range(7, 14)) / 7.0
        df["lag_mean_7_13"] = lag_7_13
    df["jour_semaine"] = df.index.dayofweek
    df["mois"] = df.index.month
    # SUPPRIMÉ : jour_du_mois et fin_mois (créent des artefacts de fin de mois non réalistes)
    # Un hôpital n'a pas de raison d'avoir moins de patients le 1er du mois
    # Saison grippe / hiver (nov–mars) et mois d'hiver stricts (déc–fév) pour que le ML apprenne les pics épidémiques
    df["saison_grippe"] = df["mois"].isin((11, 12, 1, 2, 3)).astype(int)
    df["mois_hiver"] = df["mois"].isin((12, 1, 2)).astype(int)
    if use_calendar_ferie:
        df["jour_ferie"] = np.array([1 if is_jour_ferie(d) else 0 for d in df.index])
        df["veille_ferie"] = np.array([1 if is_veille_ferie(d) else 0 for d in df.index])
        df["lendemain_ferie"] = np.array([1 if is_lendemain_ferie(d) else 0 for d in df.index])
        df["vacances_scolaires"] = np.array([1 if is_vacances_scolaires_zone_c(d) else 0 for d in df.index])
    if use_temperature:
        df["temperature"] = np.array([temperature_synthetique_paris(d) for d in df.index])
    df = df.dropna()
    if df.empty or len(df) < 30:
        return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), pd.DatetimeIndex([], freq="D")

    feature_cols = [c for c in df.columns if c != "y"]
    X_train = df[feature_cols]
    y_train = df["y"]

    last_date = series.index[-1]
    pred_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    X_pred_list = []
    for i, d in enumerate(pred_dates):
        row = {}
        for lag in [1, 7, 14]:
            key = "lag_{}".format(lag)
            if key in feature_cols:
                lag_date = d - pd.Timedelta(days=lag)
                row[key] = series.get(lag_date, np.nan)
        if "lag_mean_7_13" in feature_cols:
            vals = [series.get(d - pd.Timedelta(days=k), np.nan) for k in range(7, 14)]
            row["lag_mean_7_13"] = np.nanmean(vals) if any(np.isfinite(vals)) else np.nan
        row["jour_semaine"] = d.dayofweek
        row["mois"] = d.month
        # SUPPRIMÉ : jour_du_mois et fin_mois (artefacts)
        row["saison_grippe"] = 1 if d.month in (11, 12, 1, 2, 3) else 0
        row["mois_hiver"] = 1 if d.month in (12, 1, 2) else 0
        if use_calendar_ferie:
            row["jour_ferie"] = 1 if is_jour_ferie(d) else 0
            row["veille_ferie"] = 1 if is_veille_ferie(d) else 0
            row["lendemain_ferie"] = 1 if is_lendemain_ferie(d) else 0
            row["vacances_scolaires"] = 1 if is_vacances_scolaires_zone_c(d) else 0
        if use_temperature:
            row["temperature"] = temperature_synthetique_paris(d)
        X_pred_list.append(row)
    X_pred = pd.DataFrame(X_pred_list)
    for c in X_pred.columns:
        if X_pred[c].isna().any() and c in X_train.columns:
            X_pred[c] = X_pred[c].fillna(X_train[c].mean())
    if X_pred.isna().any().any():
        X_pred = X_pred.fillna(series.iloc[-28:].mean())
    return X_train, y_train, X_pred, pred_dates


def predict_regression(
    series: pd.Series,
    horizon_jours: int = 14,
    use_splines: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Prévision par régression Ridge sur lags (1, 7, 14) + calendrier.
    Si use_splines=True (défaut), approximation GAM par splines (Bouteloup) sur
    jour_semaine, jour_du_mois, température pour capturer des effets non linéaires.
    """
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler, SplineTransformer
    except ImportError:
        return None

    out = _build_lag_calendar_features(series, horizon_jours)
    if len(out) != 4:
        return None
    X_train, y_train, X_pred, pred_dates = out
    if X_train.empty or X_pred.empty:
        return None

    spl_cols = [c for c in ["jour_semaine", "jour_du_mois", "temperature"] if c in X_train.columns]
    if use_splines and len(spl_cols) >= 1:
        other_cols = [c for c in X_train.columns if c not in spl_cols]
        X_train_other = X_train[other_cols].copy()
        X_pred_other = X_pred[other_cols].copy()
        for c in other_cols:
            if X_pred_other[c].isna().any():
                X_pred_other[c] = X_pred_other[c].fillna(X_train_other[c].mean())
        spline = SplineTransformer(n_knots=5, degree=3)
        X_train_spl = spline.fit_transform(X_train[spl_cols])
        X_pred_spl = spline.transform(X_pred[spl_cols])
        X_train = np.hstack([X_train_other.values, X_train_spl])
        X_pred = np.hstack([X_pred_other.values, X_pred_spl])
    else:
        X_train = X_train.values
        X_pred = X_pred.values

    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train)
    Xp = scaler.transform(X_pred)
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(Xt, y_train)
    pred = np.maximum(0, model.predict(Xp))
    resid = y_train - model.predict(Xt)
    std = resid.std() or np.abs(pred).mean() * 0.1
    low = np.maximum(0, pred - 1.96 * std)
    high = pred + 1.96 * std

    return pd.DataFrame({
        "date": pred_dates,
        "prediction": pred,
        "prediction_low": low,
        "prediction_high": high,
        "type": "admissions",
    })


# --- Boosting (XGBoost ou GradientBoostingRegressor) : apprentissage sur le passé ---
def predict_boosting(
    series: pd.Series,
    horizon_jours: int = 14,
    use_splines: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Prévision par boosting (XGBoost ou GBM). Mêmes features que Ridge + option splines (type GAM).
    """
    try:
        import xgboost as xgb  # type: ignore[import-untyped]
        _use_xgb = True
    except ImportError:
        _use_xgb = False
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler, SplineTransformer
    except ImportError:
        return None

    out = _build_lag_calendar_features(series, horizon_jours)
    if len(out) != 4:
        return None
    X_train, y_train, X_pred, pred_dates = out
    if X_train.empty or X_pred.empty:
        return None

    spl_cols = [c for c in ["jour_semaine", "jour_du_mois", "temperature"] if c in X_train.columns]
    if use_splines and len(spl_cols) >= 1:
        other_cols = [c for c in X_train.columns if c not in spl_cols]
        X_train_other = X_train[other_cols].copy()
        X_pred_other = X_pred[other_cols].copy()
        for c in other_cols:
            if X_pred_other[c].isna().any():
                X_pred_other[c] = X_pred_other[c].fillna(X_train_other[c].mean())
        spline = SplineTransformer(n_knots=5, degree=3)
        X_train_spl = spline.fit_transform(X_train[spl_cols])
        X_pred_spl = spline.transform(X_pred[spl_cols])
        X_train = np.hstack([X_train_other.values, X_train_spl])
        X_pred = np.hstack([X_pred_other.values, X_pred_spl])
    else:
        X_train = X_train.values
        X_pred = X_pred.values

    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train)
    Xp = scaler.transform(X_pred)
    if _use_xgb:
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            objective="reg:squarederror",
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
    model.fit(Xt, y_train)
    pred = np.maximum(0, model.predict(Xp))
    resid = y_train - model.predict(Xt)
    std = float(resid.std() or np.abs(pred).mean() * 0.1)
    low = np.maximum(0, pred - 1.96 * std)
    high = pred + 1.96 * std

    return pd.DataFrame({
        "date": pred_dates,
        "prediction": pred,
        "prediction_low": low,
        "prediction_high": high,
        "type": "admissions",
    })


def evaluate_boosting_model(
    series: pd.Series,
    validation_days: int = 90,
) -> Dict[str, Any]:
    """
    Évalue le modèle boosting sur une période de validation (train sur le passé, test sur les derniers jours).
    Retourne MAE, RMSE, % à ±10 % (réf. Bouteloup), biais, et comparaison avec le modèle « best » (Holt-Winters/Ridge).
    """
    series = _ensure_daily_index(series)
    if len(series) < validation_days + 60:
        return {
            "mae": None,
            "rmse": None,
            "pct_within_10": None,
            "mean_error": None,
            "message": "Série trop courte pour la validation.",
            "boosting_vs_best": None,
        }
    train_end = len(series) - validation_days
    train = series.iloc[:train_end]
    test = series.iloc[train_end:]
    horizon = len(test)

    pred_boosting = predict_boosting(train, horizon_jours=horizon)
    pred_best = predict_admissions_best(train, horizon_jours=horizon)
    if pred_boosting is None or pred_boosting.empty or len(pred_boosting) < horizon:
        return {
            "mae": None,
            "rmse": None,
            "pct_within_10": None,
            "mean_error": None,
            "message": "Échec prédiction boosting.",
            "boosting_vs_best": None,
        }
    if pred_best is None or pred_best.empty or len(pred_best) < horizon:
        pred_best = predict_moving_average(train, horizon_jours=horizon)

    actual = test.values[:horizon]
    p_boost = pred_boosting["prediction"].values[:horizon]
    p_best = pred_best["prediction"].values[:horizon]

    mae = float(np.mean(np.abs(p_boost - actual)))
    rmse = float(np.sqrt(np.mean((p_boost - actual) ** 2)))
    rel_err = np.where(actual > 0, np.abs(p_boost - actual) / np.maximum(actual, 1e-6), 0)
    pct_within_10 = float((rel_err <= 0.10).mean())
    mean_error = float(np.mean(p_boost - actual))

    mae_best = float(np.mean(np.abs(p_best - actual)))
    rmse_best = float(np.sqrt(np.mean((p_best - actual) ** 2)))
    boosting_vs_best = {
        "mae_boosting": mae,
        "mae_best": mae_best,
        "rmse_boosting": rmse,
        "rmse_best": rmse_best,
        "meilleur_mae": "boosting" if mae <= mae_best else "best",
        "meilleur_rmse": "boosting" if rmse <= rmse_best else "best",
    }

    return {
        "mae": mae,
        "rmse": rmse,
        "pct_within_10": pct_within_10,
        "mean_error": mean_error,
        "n_days": horizon,
        "message": None,
        "boosting_vs_best": boosting_vs_best,
    }


# --- SARIMA (optionnel) ---
def predict_sarima(
    series: pd.Series,
    horizon_jours: int = 14,
    order: Tuple[int, int, int] = (1, 0, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 7),
) -> Optional[pd.DataFrame]:
    """
    Prévision SARIMA avec saisonnalité 7 jours. Fallback sur None si échec.
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        return None

    series = _ensure_daily_index(series)
    if len(series) < 4 * seasonal_order[3]:
        return None

    y = series.clip(lower=1e-6)
    try:
        model = SARIMAX(
            y,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False, maxiter=100)
    except Exception:
        return None

    forecast = fit.get_forecast(steps=horizon_jours)  # type: ignore[union-attr]
    # Utiliser summary_frame() pour éviter tout accès à predicted_mean/predicted_values
    # (selon version statsmodels, le wrapper peut lever AttributeError)
    try:
        sf = forecast.summary_frame(alpha=0.05)
    except Exception:
        sf = forecast.summary_frame()
    pred = np.asarray(sf["mean"]).ravel()
    if "mean_ci_lower" in sf.columns and "mean_ci_upper" in sf.columns:
        low = np.asarray(sf["mean_ci_lower"]).ravel()
        high = np.asarray(sf["mean_ci_upper"]).ravel()
    else:
        se = np.asarray(sf["mean_se"]).ravel() if "mean_se" in sf.columns else np.full_like(pred, np.sqrt(float(getattr(fit, "params", {}).get("sigma2", 1))))
        low = np.maximum(0, pred - 1.96 * se)
        high = pred + 1.96 * se

    dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon_jours, freq="D")
    return pd.DataFrame({
        "date": dates,
        "prediction": np.maximum(0, pred),
        "prediction_low": np.maximum(0, low),
        "prediction_high": high,
        "type": "admissions",
    })


# --- Benchmark : sélection du meilleur modèle par backtest ---
def select_best_model_by_backtest(
    series: pd.Series,
    validation_days: int = 90,
    metric_primary: str = "pct_within_10",
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """
    Backteste les 4 modèles (Holt-Winters, Ridge, Boosting, SARIMA) sur les validation_days
    derniers jours. Retourne le nom du meilleur modèle et les métriques par modèle.
    Critère principal : metric_primary ("pct_within_10" = plus haut mieux, "mae"/"rmse" = plus bas mieux).
    En cas d'égalité sur pct_within_10, on départage par MAE (plus bas mieux).
    """
    series = _ensure_daily_index(series)
    min_len = validation_days + 60
    if len(series) < min_len:
        return "holt_winters", {}  # fallback, pas assez de données pour benchmark

    train_end = len(series) - validation_days
    train = series.iloc[:train_end]
    test = series.iloc[train_end:]
    horizon = len(test)
    actual = test.values

    models_to_test = [
        ("holt_winters", predict_holt_winters),
        ("regression", predict_regression),
        ("boosting", predict_boosting),
        ("sarima", predict_sarima),
    ]
    results: Dict[str, Dict[str, float]] = {}

    for name, fn in models_to_test:
        try:
            pred_df = fn(train, horizon_jours=horizon)
        except Exception:
            continue
        if pred_df is None or pred_df.empty or len(pred_df) < horizon:
            continue
        pred = pred_df["prediction"].values[:horizon]
        mae = float(np.mean(np.abs(pred - actual)))
        rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
        rel_err = np.where(actual > 0, np.abs(pred - actual) / np.maximum(actual, 1e-6), 0)
        pct_within_10 = float((rel_err <= 0.10).mean())
        results[name] = {"mae": mae, "rmse": rmse, "pct_within_10": pct_within_10}

    if not results:
        return "holt_winters", {}

    # Meilleur : d'abord par pct_within_10 (desc), puis par MAE (asc)
    def score(name: str) -> Tuple[float, float]:
        r = results[name]
        return (-r["pct_within_10"], r["mae"])

    best_name = min(results.keys(), key=score)
    return best_name, results


# --- Choix du meilleur modèle (admissions) ---
def predict_admissions_best(
    series: pd.Series,
    horizon_jours: int = 14,
    prefer: str = "holt_winters_stable",  # FORCÉ : utiliser le HW stable par défaut
) -> pd.DataFrame:
    """
    Prévision des admissions.
    CORRECTION 5 FÉV 2026 : Par défaut, on utilise Holt-Winters avec niveau stable
    pour éviter les dérives descendantes irréalistes des modèles de régression.
    - prefer="holt_winters_stable" (défaut) : HW avec moyenne stable + saisonnalité hebdo
    - prefer="best_by_backtest" : benchmark des 4 modèles, utilise le meilleur (peut dériver)
    - Sinon : cascade (premier qui réussit)
    """
    series = _ensure_daily_index(series)
    if series.empty or len(series) < 7:
        return predict_moving_average(series, horizon_jours=horizon_jours)

    # PAR DÉFAUT : utiliser Holt-Winters stable (pas de dérive)
    if prefer == "holt_winters_stable" or prefer == "holt_winters":
        result = predict_holt_winters(series, horizon_jours=horizon_jours)
        if result is not None and len(result) >= horizon_jours:
            return result.head(horizon_jours) if hasattr(result, "head") else result
        # Fallback si HW échoue
        prefer = "ma"

    # Sélection par benchmark : on choisit le modèle le plus "true to real" (meilleur % ±10 %, puis MAE)
    if prefer == "best_by_backtest":
        validation_days = min(90, max(28, (len(series) // 3)))
        best_name, _ = select_best_model_by_backtest(series, validation_days=validation_days)
        model_map = {
            "holt_winters": predict_holt_winters,
            "regression": predict_regression,
            "boosting": predict_boosting,
            "sarima": predict_sarima,
        }
        if best_name in model_map:
            try:
                result = model_map[best_name](series, horizon_jours=horizon_jours)
                if result is not None and not (hasattr(result, "empty") and result.empty) and len(result) >= horizon_jours:
                    return result.head(horizon_jours) if hasattr(result, "head") else result
            except Exception:
                pass
        # Si le meilleur échoue à l'inférence (ex. série trop courte pour SARIMA), on enchaîne la cascade

    # Cascade par défaut : HW d'abord (stable)
    order = [
        ("holt_winters", predict_holt_winters),
        ("regression", predict_regression),
        ("boosting", predict_boosting),
        ("sarima", predict_sarima),
    ]
    if prefer == "regression":
        order = [
            ("regression", predict_regression),
            ("holt_winters", predict_holt_winters),
            ("boosting", predict_boosting),
            ("sarima", predict_sarima),
        ]
    elif prefer == "boosting":
        order = [
            ("boosting", predict_boosting),
            ("regression", predict_regression),
            ("holt_winters", predict_holt_winters),
            ("sarima", predict_sarima),
        ]
    elif prefer == "sarima":
        order = [
            ("sarima", predict_sarima),
            ("holt_winters", predict_holt_winters),
            ("regression", predict_regression),
            ("boosting", predict_boosting),
        ]

    for name, fn in order:
        try:
            if name == "sarima":
                result = fn(series, horizon_jours=horizon_jours)
            else:
                result = fn(series, horizon_jours=horizon_jours)
        except Exception:
            result = None
        if result is not None and not (hasattr(result, "empty") and result.empty) and len(result) >= horizon_jours:
            return result.head(horizon_jours) if hasattr(result, "head") else result

    return predict_moving_average(series, horizon_jours=horizon_jours)


# --- Occupation : prédiction directe (série occupation_lits) ---
def predict_occupation_direct(
    occupation_df: pd.DataFrame,
    horizon_jours: int = 14,
    prefer: str = "auto",
) -> pd.DataFrame:
    """
    Prévision directe du nombre de lits occupés (Holt-Winters ou régression, puis MA).
    prefer : "auto" (essayer HW puis Ridge puis MA), "holt_winters" (HW uniquement), "regression" (Ridge uniquement).
    """
    occ = prepare_series(occupation_df, "occupation_lits")
    occ = _ensure_daily_index(occ)
    if occ.empty:
        return pd.DataFrame()

    if prefer != "regression":
        result = predict_holt_winters(occ, horizon_jours=horizon_jours)
        if result is not None and len(result) == horizon_jours:
            result = result.rename(columns={
                "prediction": "occupation_lits_pred",
                "prediction_low": "occupation_lits_low",
                "prediction_high": "occupation_lits_high",
            })
            result["admissions_pred"] = np.nan  # non utilisé en mode direct
            return result[["date", "occupation_lits_pred", "occupation_lits_low", "occupation_lits_high", "admissions_pred"]]

    if prefer != "holt_winters":
        result = predict_regression(occ, horizon_jours=horizon_jours)
        if result is not None and len(result) == horizon_jours:
            result = result.rename(columns={
                "prediction": "occupation_lits_pred",
                "prediction_low": "occupation_lits_low",
                "prediction_high": "occupation_lits_high",
            })
            result["admissions_pred"] = np.nan
            return result[["date", "occupation_lits_pred", "occupation_lits_low", "occupation_lits_high", "admissions_pred"]]

    # Fallback : moyenne récente (tendance = 0 pour éviter dérive)
    window = 28
    last = occ.iloc[-window:].mean()
    std = occ.iloc[-window:].std() or last * 0.05
    dates = pd.date_range(occ.index[-1] + pd.Timedelta(days=1), periods=horizon_jours, freq="D")
    rows = []
    for i, d in enumerate(dates):
        val = max(0, last)
        rows.append({
            "date": d,
            "occupation_lits_pred": val,
            "occupation_lits_low": max(0, val - 1.96 * std),
            "occupation_lits_high": val + 1.96 * std,
            "admissions_pred": np.nan,
        })
    return pd.DataFrame(rows)


# --- Durée de séjour saisonnière (réf. Lequertier : la durée de séjour varie) ---
def _duree_sejour_saisonniere(month: int, base: float = 6.0) -> float:
    """
    Durée de séjour moyenne selon le mois (réf. Lequertier).
    Variation continue (sinusoïdale) pour éviter des sauts brutaux entre mois.
    Hiver (déc-janv) : pic ; été (juin-juil) : creux.
    """
    # Variation sinusoïdale : max en janvier (mois 1), min en juillet (mois 7)
    # Amplitude réduite à ±4% (au lieu de ±8%) pour plus de stabilité
    import math
    angle = 2 * math.pi * (month - 1) / 12.0  # 0 à 2π sur l'année
    # Déphasage pour avoir le max en janvier : cos(0) = 1
    variation = 0.04 * math.cos(angle)  # ±4%
    return base * (1.0 + variation)


# --- Occupation à partir des admissions (modèle stock + durée séjour) ---
def predict_occupation_from_admissions(
    occupation_df: pd.DataFrame,
    horizon_jours: int = 14,
    duree_sejour_moy: Optional[float] = None,
    use_best_admissions: bool = True,
    duree_sejour_saisonniere: bool = True,
    pred_admissions_df: Optional[pd.DataFrame] = None,
    pred_adm: Optional[pd.DataFrame] = None,
    capacite_lits: int = 1800,
) -> pd.DataFrame:
    """
    Prévision de l'occupation à partir des admissions prédites via un VRAI modèle stock-flux.
    
    Modèle dynamique :
        Stock(t) = min(Stock(t-1) + Entrées(t) - Sorties(t), Capacité)
        Sorties(t) = Stock(t-1) / DMS(t)  (durée moyenne de séjour)
    
    Si pred_admissions_df (ou pred_adm) est fourni, il est utilisé tel quel (colonnes date, prediction, prediction_low, prediction_high).
    Sinon : use_best_admissions=True appelle predict_admissions_best, False appelle MA.
    
    duree_sejour_moy : DMS en jours. Si None, calculée automatiquement à partir des données historiques (Stock moyen / Admissions moyennes).
    capacite_lits : capacité maximale (plafonnement du stock pour rester réaliste).
    """
    # Alias pour compatibilité (certains appels utilisent pred_adm)
    if pred_admissions_df is None and pred_adm is not None:
        pred_admissions_df = pred_adm
    occ = prepare_series(occupation_df, "occupation_lits")
    adm = prepare_series(occupation_df, "admissions_jour")
    occ = _ensure_daily_index(occ)
    adm = _ensure_daily_index(adm)

    # Calcul automatique de la DMS si non fournie (régime permanent : Stock = Admissions × DMS)
    if duree_sejour_moy is None:
        occ_mean_hist = occ.iloc[-90:].mean() if len(occ) >= 90 else occ.mean()
        adm_mean_hist = adm.iloc[-90:].mean() if len(adm) >= 90 else adm.mean()
        duree_sejour_moy = occ_mean_hist / adm_mean_hist if adm_mean_hist > 0 else 4.0
        # Plafonner entre 2 et 10 jours pour rester réaliste
        duree_sejour_moy = max(2.0, min(10.0, float(duree_sejour_moy)))
    dms: float = duree_sejour_moy if duree_sejour_moy is not None else 6.0

    if pred_admissions_df is not None and len(pred_admissions_df) >= horizon_jours:
        pred_adm = pred_admissions_df.head(horizon_jours).copy()
        if "prediction" not in pred_adm.columns:
            pred_adm = pred_adm.rename(columns={pred_adm.columns[1]: "prediction"}) if len(pred_adm.columns) > 1 else pred_adm
    else:
        if use_best_admissions:
            pred_adm = predict_admissions_best(adm, horizon_jours=horizon_jours)
        else:
            pred_adm = predict_moving_average(adm, horizon_jours=horizon_jours)

    # Initialisation : stock actuel (dernier jour observé)
    stock_actuel_pred = float(occ.iloc[-1])
    # IC réduits : ±3% au lieu de ±5% pour éviter l'accumulation jusqu'à 100%
    stock_actuel_low = stock_actuel_pred * 0.97
    stock_actuel_high = stock_actuel_pred * 1.03
    
    # INERTIE : Pour éviter des changements trop brutaux, on ajoute un lissage exponentiel
    # alpha = 0.3 : le nouveau stock est 70% du stock précédent + 30% du calcul brut
    # Cela correspond à la réalité : un hôpital change progressivement (sorties étalées sur plusieurs jours)
    alpha_inertie = 0.3  # Poids du nouveau calcul (30% = inertie modérée)

    pred = []
    for _, row in pred_adm.iterrows():
        # Entrées prédites (admissions)
        entrees_pred = float(row.get("prediction", 0) or 0)
        # IC admissions : ±8% au lieu de ±10% (évite accumulation excessive sur l'occupation)
        entrees_low = float(row.get("prediction_low", entrees_pred * 0.92) or 0)
        entrees_high = float(row.get("prediction_high", entrees_pred * 1.08) or 0)
        
        d = row["date"]
        d_ts = pd.Timestamp(d) if not isinstance(d, pd.Timestamp) else d
        month = int(d_ts.month)
        
        # Durée de séjour saisonnière (variation continue)
        duree = _duree_sejour_saisonniere(month, dms) if duree_sejour_saisonniere else dms
        
        # Sorties = Stock / DMS (modèle exponentiel)
        sorties_pred = stock_actuel_pred / duree
        sorties_low = stock_actuel_low / duree
        sorties_high = stock_actuel_high / duree
        
        # Nouveau stock = Stock précédent + Entrées - Sorties (plafonné à la capacité)
        stock_brut_pred = stock_actuel_pred + entrees_pred - sorties_pred
        stock_brut_low = stock_actuel_low + entrees_low - sorties_low
        stock_brut_high = stock_actuel_high + entrees_high - sorties_high
        
        # Lissage exponentiel pour l'inertie (changements progressifs)
        # CORRECTION : Plafonner à capacite_lits (1800), PAS 110% (physiquement impossible)
        stock_actuel_pred = min(capacite_lits, max(0, (1 - alpha_inertie) * stock_actuel_pred + alpha_inertie * stock_brut_pred))
        stock_actuel_low = min(capacite_lits, max(0, (1 - alpha_inertie) * stock_actuel_low + alpha_inertie * stock_brut_low))
        # IC haut : plafonnement à 95% de la capacité (1710 lits) pour éviter affichage "100%" irréaliste
        stock_actuel_high = min(capacite_lits * 0.95, max(0, (1 - alpha_inertie) * stock_actuel_high + alpha_inertie * stock_brut_high))
        
        pred.append({
            "date": d_ts,
            "occupation_lits_pred": stock_actuel_pred,
            "occupation_lits_low": stock_actuel_low,
            "occupation_lits_high": stock_actuel_high,
            "admissions_pred": entrees_pred,
        })

    return pd.DataFrame(pred)


# --- Choix prédiction occupation : directe vs via admissions ---
def predict_occupation_best(
    occupation_df: pd.DataFrame,
    horizon_jours: int = 14,
    capacite_lits: int = 1800,
    duree_sejour_moy: float = 6.0,
) -> pd.DataFrame:
    """
    Prévision de l'occupation : prédiction directe (Holt-Winters/régression sur occupation)
    si disponible, sinon via admissions. Ajoute taux_occupation et intervalles.
    """
    direct = predict_occupation_direct(occupation_df, horizon_jours=horizon_jours)
    if not direct.empty and direct["occupation_lits_pred"].notna().all():
        direct["taux_occupation_pred"] = direct["occupation_lits_pred"] / capacite_lits
        direct["taux_occupation_low"] = direct["occupation_lits_low"] / capacite_lits
        direct["taux_occupation_high"] = direct["occupation_lits_high"] / capacite_lits
        return direct

    from_adm = predict_occupation_from_admissions(
        occupation_df, horizon_jours=horizon_jours, duree_sejour_moy=duree_sejour_moy, use_best_admissions=True, capacite_lits=capacite_lits
    )
    from_adm["taux_occupation_pred"] = from_adm["occupation_lits_pred"] / capacite_lits
    from_adm["taux_occupation_low"] = from_adm["occupation_lits_low"] / capacite_lits
    from_adm["taux_occupation_high"] = from_adm["occupation_lits_high"] / capacite_lits
    return from_adm


def build_besoins_from_occupation_pred(pred_df: pd.DataFrame, capacite_lits: int = 1800) -> Dict[str, Any]:
    """À partir d'un DataFrame de prévisions (occupation_lits_pred, low, high), ajoute taux, alerte, et retourne le dict besoins (public, pour le dashboard)."""
    return _besoins_from_pred_df(pred_df, capacite_lits)


def _besoins_from_pred_df(pred_df: pd.DataFrame, capacite_lits: int) -> Dict[str, Any]:
    """À partir d'un DataFrame de prévisions (occupation_lits_pred, low, high), ajoute taux, alerte, et retourne le dict besoins."""
    if pred_df.empty:
        return {
            "previsions": pred_df,
            "taux_max_prevu": 0.0,
            "taux_max_high": 0.0,
            "recommandation": "Données insuffisantes.",
            "seuils": {"alerte": 0.85, "critique": 0.95},
        }
    if "taux_occupation_pred" not in pred_df.columns:
        pred_df = pred_df.copy()
        pred_df["taux_occupation_pred"] = pred_df["occupation_lits_pred"] / capacite_lits
        if "occupation_lits_low" in pred_df.columns:
            pred_df["taux_occupation_low"] = pred_df["occupation_lits_low"] / capacite_lits
        else:
            pred_df["taux_occupation_low"] = pred_df["taux_occupation_pred"] * 0.95
        if "occupation_lits_high" in pred_df.columns:
            pred_df["taux_occupation_high"] = pred_df["occupation_lits_high"] / capacite_lits
        else:
            pred_df["taux_occupation_high"] = pred_df["taux_occupation_pred"] * 1.05
    seuil_alerte, seuil_critique = 0.85, 0.95
    pred_df = pred_df.copy()
    pred_df["alerte"] = "normal"
    pred_df.loc[pred_df["taux_occupation_pred"] >= seuil_critique, "alerte"] = "critique"
    pred_df.loc[
        (pred_df["taux_occupation_pred"] >= seuil_alerte) & (pred_df["taux_occupation_pred"] < seuil_critique),
        "alerte",
    ] = "alerte"
    max_occ = float(pred_df["taux_occupation_pred"].max())
    max_high = float(pred_df["taux_occupation_high"].max()) if "taux_occupation_high" in pred_df.columns else max_occ
    if max_occ >= seuil_critique:
        reco = "Renforcer les effectifs et reporter les interventions non urgentes."
    elif max_occ >= seuil_alerte:
        reco = "Surveiller les effectifs et préparer une montée en charge."
    elif max_high >= seuil_alerte:
        reco = "Vigilance : la borne haute des prévisions approche le seuil d'alerte."
    else:
        reco = "Capacité dans la norme ; maintenir la vigilance."
    return {
        "previsions": pred_df,
        "taux_max_prevu": max_occ,
        "taux_max_high": max_high,
        "recommandation": reco,
        "seuils": {"alerte": seuil_alerte, "critique": seuil_critique},
    }


def predict_besoins(
    occupation_df: pd.DataFrame,
    capacite_lits: int = 1800,
    horizon_jours: int = 14,
    duree_sejour_moy: float = 6.0,
) -> Dict[str, Any]:
    """
    Prévision des besoins (lits, alerte) avec modèles améliorés.
    Retourne prévisions (avec intervalles si dispo), taux max, recommandation, seuils.
    """
    pred_df = predict_occupation_best(
        occupation_df, horizon_jours=horizon_jours, capacite_lits=capacite_lits, duree_sejour_moy=duree_sejour_moy
    )
    if pred_df.empty:
        pred_df = predict_occupation_from_admissions(occupation_df, horizon_jours=horizon_jours, capacite_lits=capacite_lits)
        pred_df["taux_occupation_pred"] = pred_df["occupation_lits_pred"] / capacite_lits
        pred_df["taux_occupation_low"] = pred_df["taux_occupation_pred"] * 0.95
        pred_df["taux_occupation_high"] = pred_df["taux_occupation_pred"] * 1.05
    return _besoins_from_pred_df(pred_df, capacite_lits)


def predict_besoins_with_model(
    occupation_df: pd.DataFrame,
    model_choice: str,
    capacite_lits: int = 1800,
    horizon_jours: int = 14,
    duree_sejour_moy: float = 6.0,
) -> Dict[str, Any]:
    """
    Prévision des besoins en forçant un modèle donné (pour comparaison et démo).
    model_choice : "auto" | "holt_winters" | "ridge" | "sarima" | "ma" | "boosting" | "direct_hw" | "direct_ridge"
    """
    occ = prepare_series(occupation_df, "occupation_lits")
    adm = prepare_series(occupation_df, "admissions_jour")
    occ = _ensure_daily_index(occ)
    adm = _ensure_daily_index(adm)

    if model_choice == "auto":
        return predict_besoins(
            occupation_df, capacite_lits=capacite_lits, horizon_jours=horizon_jours, duree_sejour_moy=duree_sejour_moy
        )

    if model_choice == "direct_hw":
        pred_df = predict_occupation_direct(occupation_df, horizon_jours=horizon_jours, prefer="holt_winters")
        if not pred_df.empty:
            pred_df["taux_occupation_pred"] = pred_df["occupation_lits_pred"] / capacite_lits
            pred_df["taux_occupation_low"] = pred_df["occupation_lits_low"] / capacite_lits
            pred_df["taux_occupation_high"] = pred_df["occupation_lits_high"] / capacite_lits
        return _besoins_from_pred_df(pred_df, capacite_lits)

    if model_choice == "direct_ridge":
        pred_df = predict_occupation_direct(occupation_df, horizon_jours=horizon_jours, prefer="regression")
        if not pred_df.empty:
            pred_df["taux_occupation_pred"] = pred_df["occupation_lits_pred"] / capacite_lits
            pred_df["taux_occupation_low"] = pred_df["occupation_lits_low"] / capacite_lits
            pred_df["taux_occupation_high"] = pred_df["occupation_lits_high"] / capacite_lits
        return _besoins_from_pred_df(pred_df, capacite_lits)

    pred_adm = None
    if model_choice == "holt_winters":
        pred_adm = predict_holt_winters(adm, horizon_jours=horizon_jours)
    elif model_choice == "ridge":
        pred_adm = predict_regression(adm, horizon_jours=horizon_jours)
    elif model_choice == "sarima":
        pred_adm = predict_sarima(adm, horizon_jours=horizon_jours)
    elif model_choice == "ma":
        pred_adm = predict_moving_average(adm, horizon_jours=horizon_jours)
    elif model_choice == "boosting":
        pred_adm = predict_boosting(adm, horizon_jours=horizon_jours)

    if pred_adm is None or pred_adm.empty or len(pred_adm) < horizon_jours:
        return predict_besoins(
            occupation_df, capacite_lits=capacite_lits, horizon_jours=horizon_jours, duree_sejour_moy=duree_sejour_moy
        )
    pred_df = predict_occupation_from_admissions(
        occupation_df,
        horizon_jours=horizon_jours,
        duree_sejour_moy=duree_sejour_moy,
        pred_admissions_df=pred_adm,
        capacite_lits=capacite_lits,
    )
    return _besoins_from_pred_df(pred_df, capacite_lits)


# --- Métrique de validation ±10 % (réf. Bouteloup 2020) ---
def evaluate_forecast_pct_within_10(
    series: pd.Series,
    validation_days: int = 90,
    use_best: bool = True,
) -> Dict[str, Any]:
    """
    Évalue le modèle sur une période de validation : % de jours à ±10 % (réf. Bouteloup)
    et biais moyen (sous- vs surestimation). Train sur series[:-validation_days], prédit les validation_days suivants.
    """
    series = _ensure_daily_index(series)
    if len(series) < validation_days + 60:
        return {"pct_within_10": None, "mean_error": None, "n_days": 0, "message": "Série trop courte"}
    train_end = len(series) - validation_days
    train = series.iloc[:train_end]
    test = series.iloc[train_end:]
    horizon = len(test)
    if use_best:
        pred_df = predict_admissions_best(train, horizon_jours=horizon)
    else:
        pred_df = predict_moving_average(train, horizon_jours=horizon)
    if pred_df is None or pred_df.empty or len(pred_df) < horizon:
        return {"pct_within_10": None, "mean_error": None, "n_days": 0, "message": "Échec prédiction"}
    pred = pred_df["prediction"].values[:horizon]
    actual = test.values[:horizon]
    # Éviter division par zéro
    rel_err = np.where(actual > 0, np.abs(pred - actual) / np.maximum(actual, 1e-6), 0)
    within_10 = (rel_err <= 0.10).mean()
    mean_error = float(np.mean(pred - actual))  # positif = surestimation, négatif = sous-estimation
    return {
        "pct_within_10": float(within_10),
        "mean_error": mean_error,
        "n_days": horizon,
        "pct_surestimation": float((pred > actual * 1.10).mean()),
        "pct_sous_estimation": float((pred < actual * 0.90).mean()),
    }


def run_backtest_admissions(
    series: pd.Series,
    validation_days: int = 90,
    use_best: bool = True,
) -> Dict[str, Any]:
    """
    Backtest : entraîne sur le passé, prédit la période de validation, retourne
    un DataFrame prévision vs réel et les métriques (MAE, RMSE, % ±10 %, biais).
    """
    series = _ensure_daily_index(series)
    if len(series) < validation_days + 60:
        return {
            "backtest_df": pd.DataFrame(),
            "metrics": None,
            "message": "Série trop courte (il faut au moins 60 jours avant la période de test).",
        }
    train_end = len(series) - validation_days
    train = series.iloc[:train_end]
    test = series.iloc[train_end:]
    horizon = len(test)

    if use_best:
        pred_df = predict_admissions_best(train, horizon_jours=horizon)
    else:
        pred_df = predict_moving_average(train, horizon_jours=horizon)

    if pred_df is None or pred_df.empty or len(pred_df) < horizon:
        return {
            "backtest_df": pd.DataFrame(),
            "metrics": None,
            "message": "Échec de la prédiction pour le backtest.",
        }

    actual = test.values[:horizon]
    pred = pred_df["prediction"].values[:horizon]
    dates = pred_df["date"].values[:horizon]
    dates = pd.to_datetime(dates)

    backtest_df = pd.DataFrame({
        "date": dates,
        "observé": actual,
        "prévu": pred,
    })
    if "prediction_low" in pred_df.columns and "prediction_high" in pred_df.columns:
        backtest_df["prévu_basse"] = pred_df["prediction_low"].values[:horizon]
        backtest_df["prévu_haute"] = pred_df["prediction_high"].values[:horizon]

    rel_err = np.where(actual > 0, np.abs(pred - actual) / np.maximum(actual, 1e-6), 0)
    within_10 = float((rel_err <= 0.10).mean())
    mean_error = float(np.mean(pred - actual))
    mae = float(np.mean(np.abs(pred - actual)))
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))

    metrics = {
        "pct_within_10": within_10,
        "mean_error": mean_error,
        "mae": mae,
        "rmse": rmse,
        "n_days": horizon,
        "pct_surestimation": float((pred > actual * 1.10).mean()),
        "pct_sous_estimation": float((pred < actual * 0.90).mean()),
    }

    return {
        "backtest_df": backtest_df,
        "metrics": metrics,
        "message": None,
    }


# ==============================================================================
# SYSTÈME D'ENSEMBLE (ROBUSTESSE)
# ==============================================================================

def _evaluate_model_simple(func_model, series_train: pd.Series, series_test: pd.Series, horizon: int) -> float:
    """Helper interne pour évaluer rapidement un modèle sur le jeu de test (Score % within 10%)."""
    try:
        pred_df = func_model(series_train, horizon_jours=horizon)
        if pred_df is None or pred_df.empty or len(pred_df) < horizon:
            return 0.0
        pred = pred_df["prediction"].values[:horizon]
        actual = series_test.values[:horizon]
        rel_err = np.where(actual > 0, np.abs(pred - actual) / np.maximum(actual, 1e-6), 0)
        return float((rel_err <= 0.10).mean())
    except Exception:
        return 0.0


def predict_admissions_ensemble(series: pd.Series, horizon_jours: int = 14) -> pd.DataFrame:
    """
    PRÉVISION ROBUSTE PAR ENSEMBLE PONDÉRÉ.
    Agrège Holt-Winters, Ridge, Boosting et SARIMA selon leurs performances récentes (28 jours).
    """
    series = _ensure_daily_index(series)
    validation_days = 28
    if len(series) < validation_days + 60:
        return predict_holt_winters(series, horizon_jours=horizon_jours)

    train = series.iloc[:-validation_days]
    test = series.iloc[-validation_days:]

    candidats = [
        ("Holt-Winters", predict_holt_winters),
        ("Ridge Regression", predict_regression),
        ("Boosting", predict_boosting),
        ("SARIMA", predict_sarima)
    ]

    # 1. Calcul des scores (Poids)
    performances = []
    for nom, func in candidats:
        score = _evaluate_model_simple(func, train, test, validation_days)
        performances.append({"nom": nom, "func": func, "score": score})

    top3 = sorted(performances, key=lambda x: x["score"], reverse=True)[:3]
    total_score = sum(p["score"] for p in top3)

    if total_score <= 0.01:
        return predict_holt_winters(series, horizon_jours=horizon_jours)

    weights = {p["nom"]: p["score"] / total_score for p in top3}

    # 2. Prédiction Pondérée
    preds_futur = []
    model_preds_raw = []

    for p in top3:
        try:
            res = p["func"](series, horizon_jours=horizon_jours)
            if res is not None and len(res) >= horizon_jours:
                vals = res["prediction"].values[:horizon_jours]
                preds_futur.append(vals * weights[p["nom"]])
                model_preds_raw.append(vals)
        except Exception:
            pass

    if not preds_futur:
        return predict_holt_winters(series, horizon_jours=horizon_jours)

    final_pred = np.sum(preds_futur, axis=0)

    # 3. Intervalle de confiance basé sur la divergence des modèles + incertitude de base
    arr_preds = np.array(model_preds_raw)
    std_inter_model = np.std(arr_preds, axis=0) if arr_preds.shape[0] > 1 else (final_pred * 0.10)
    std_total = std_inter_model + (final_pred * 0.05)  # +5% incertitude intrinsèque

    dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon_jours, freq="D")
    ic_low = np.maximum(0, final_pred - 1.96 * std_total)
    ic_high = final_pred + 1.96 * std_total

    # ==========================================================================
    # POST-PROCESSING : RÉINJECTION DE LA SAISONNALITÉ HEBDOMADAIRE (VITAL)
    # L'ensemble a tendance à trop lisser. On réapplique le "pouls" de l'hôpital.
    # ==========================================================================
    # Facteurs : Lundi très fort (1.12), décroissance, Week-end calme (0.85)
    # 0=Lundi, 6=Dimanche
    coeffs_hebdo = np.array([1.12, 1.05, 1.00, 1.00, 1.02, 0.90, 0.85])

    # On récupère le jour de la semaine pour chaque date prédite
    jours_semaine = dates.dayofweek  # Index 0 à 6

    # On applique le coefficient correspondant à chaque jour
    ajustement = coeffs_hebdo[jours_semaine]

    # On module la prédiction finale (et les bornes)
    final_pred = final_pred * ajustement
    ic_low = ic_low * ajustement
    ic_high = ic_high * ajustement
    # ==========================================================================

    return pd.DataFrame({
        "date": dates,
        "prediction": final_pred,
        "prediction_low": ic_low,
        "prediction_high": ic_high,
        "type": "admissions"
    })


def fit_and_predict_admissions(
    admissions_df: pd.DataFrame,
    service: Optional[str] = None,
    horizon_jours: int = 14,
    use_best: bool = True,
) -> pd.DataFrame:
    """
    Prévision des admissions (global ou par service). use_best=True : Holt-Winters/régression.
    """
    if service:
        series = (
            admissions_df[admissions_df["service"] == service]
            .groupby("date")["admissions"]
            .sum()
        )
    else:
        series = admissions_df.groupby("date")["admissions"].sum()
    series = _ensure_daily_index(cast(pd.Series, series))
    if use_best:
        return predict_admissions_best(series, horizon_jours=horizon_jours)
    return predict_moving_average(series, horizon_jours=horizon_jours)
