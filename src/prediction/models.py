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

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

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

# Préparation des séries
def prepare_series(occupation_df: pd.DataFrame, col: str = "admissions_jour") -> pd.Series:
    """Série temporelle quotidienne avec index date."""
    df = occupation_df[["date", col]].drop_duplicates("date").sort_values("date")
    return df.set_index("date")[col]


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
        trend = (series.iloc[-7:].mean() - series.iloc[-window:-7].mean()) / 7 if window > 7 else 0
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
    Prévision par lissage exponentiel de Holt-Winters (saisonnalité additive).
    Saisonnalité hebdomadaire (7 jours). Retourne prediction + intervalles si disponibles.
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError:
        return None

    series = _ensure_daily_index(series)
    if len(series) < 2 * seasonal_period:
        return None

    # Éviter les valeurs nulles ou négatives
    y = series.clip(lower=1e-6)
    model = ExponentialSmoothing(
        y,
        seasonal_periods=seasonal_period,
        trend="add",
        seasonal="add",
        initialization_method="estimated",
    )
    try:
        fit = model.fit(optimized=True, remove_bias=False)
    except Exception:
        return None

    forecast = fit.forecast(steps=horizon_jours)
    # Intervalles approximatifs (écart-type des résidus)
    resid = y - fit.fittedvalues
    resid_std = resid.std() if len(resid) > 0 and resid.notna().any() else (y.std() or 1)
    low = np.maximum(0, forecast - 1.96 * resid_std)
    high = forecast + 1.96 * resid_std

    dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon_jours, freq="D")
    return pd.DataFrame({
        "date": dates,
        "prediction": np.maximum(0, forecast),
        "prediction_low": np.maximum(0, low),
        "prediction_high": high,
        "type": "admissions",
    })


# --- Régression : lags + calendrier (réf. Bouteloup : lags 7–13, jours fériés, vacances, météo) ---
def _build_lag_calendar_features(series: pd.Series, horizon: int, use_calendar_ferie: bool = True, use_temperature: bool = True) -> Tuple:
    """
    Construit les features pour X_train et X_pred.
    Lags 1, 7, 14 + mean(j-7 à j-13) (Bouteloup) + jour_semaine, mois + jours fériés, vacances (Bouteloup) + température synthétique (optionnel).
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
) -> Optional[pd.DataFrame]:
    """
    Prévision par régression Ridge sur lags (1, 7, 14) + jour_semaine + mois.
    Retourne prediction + intervalles approximatifs (écart-type des résidus).
    """
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None

    out = _build_lag_calendar_features(series, horizon_jours)
    if len(out) != 4:
        return None
    X_train, y_train, X_pred, pred_dates = out
    if X_train.empty or X_pred.empty:
        return None

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
) -> Optional[pd.DataFrame]:
    """
    Prévision par modèle de type boosting (XGBoost si dispo, sinon GradientBoostingRegressor).
    Mêmes features que la régression Ridge (lags + calendrier). Apprend du passé.
    """
    try:
        import xgboost as xgb
        _use_xgb = True
    except ImportError:
        _use_xgb = False
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None

    out = _build_lag_calendar_features(series, horizon_jours)
    if len(out) != 4:
        return None
    X_train, y_train, X_pred, pred_dates = out
    if X_train.empty or X_pred.empty:
        return None

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

    forecast = fit.get_forecast(steps=horizon_jours)
    pred = forecast.predicted_values.values
    try:
        ci = forecast.conf_int(alpha=0.05)
        low = ci.iloc[:, 0].values
        high = ci.iloc[:, 1].values
    except Exception:
        low = np.maximum(0, pred - 1.96 * np.sqrt(fit.params.get("sigma2", pred.var())))
        high = pred + 1.96 * np.sqrt(fit.params.get("sigma2", pred.var()))

    dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon_jours, freq="D")
    return pd.DataFrame({
        "date": dates,
        "prediction": np.maximum(0, pred),
        "prediction_low": np.maximum(0, low),
        "prediction_high": high,
        "type": "admissions",
    })


# --- Choix du meilleur modèle (admissions) ---
def predict_admissions_best(
    series: pd.Series,
    horizon_jours: int = 14,
    prefer: str = "holt_winters",
) -> pd.DataFrame:
    """
    Prévision des admissions : essaie Holt-Winters, puis régression, puis SARIMA, puis moyenne glissante.
    prefer : "holt_winters" | "regression" | "sarima" | "ma"
    """
    series = _ensure_daily_index(series)
    if series.empty or len(series) < 7:
        return predict_moving_average(series, horizon_jours=horizon_jours)

    order = [("holt_winters", predict_holt_winters), ("regression", predict_regression), ("sarima", predict_sarima)]
    if prefer == "regression":
        order = [("regression", predict_regression), ("holt_winters", predict_holt_winters), ("sarima", predict_sarima)]
    elif prefer == "sarima":
        order = [("sarima", predict_sarima), ("holt_winters", predict_holt_winters), ("regression", predict_regression)]

    for name, fn in order:
        if name == "sarima":
            result = fn(series, horizon_jours=horizon_jours)
        else:
            result = fn(series, horizon_jours=horizon_jours)
        if result is not None and len(result) == horizon_jours:
            return result

    return predict_moving_average(series, horizon_jours=horizon_jours)


# --- Occupation : prédiction directe (série occupation_lits) ---
def predict_occupation_direct(
    occupation_df: pd.DataFrame,
    horizon_jours: int = 14,
) -> pd.DataFrame:
    """
    Prévision directe du nombre de lits occupés (Holt-Winters ou régression, puis MA).
    """
    occ = prepare_series(occupation_df, "occupation_lits")
    occ = _ensure_daily_index(occ)
    if occ.empty:
        return pd.DataFrame()

    result = predict_holt_winters(occ, horizon_jours=horizon_jours)
    if result is not None and len(result) == horizon_jours:
        result = result.rename(columns={
            "prediction": "occupation_lits_pred",
            "prediction_low": "occupation_lits_low",
            "prediction_high": "occupation_lits_high",
        })
        result["admissions_pred"] = np.nan  # non utilisé en mode direct
        return result[["date", "occupation_lits_pred", "occupation_lits_low", "occupation_lits_high", "admissions_pred"]]

    result = predict_regression(occ, horizon_jours=horizon_jours)
    if result is not None and len(result) == horizon_jours:
        result = result.rename(columns={
            "prediction": "occupation_lits_pred",
            "prediction_low": "occupation_lits_low",
            "prediction_high": "occupation_lits_high",
        })
        result["admissions_pred"] = np.nan
        return result[["date", "occupation_lits_pred", "occupation_lits_low", "occupation_lits_high", "admissions_pred"]]

    # Fallback : moyenne récente + tendance
    window = 28
    last = occ.iloc[-window:].mean()
    trend = (occ.iloc[-7:].mean() - occ.iloc[-window:-7].mean()) / 7 if len(occ) >= window and window > 7 else 0
    std = occ.iloc[-window:].std() or last * 0.05
    dates = pd.date_range(occ.index[-1] + pd.Timedelta(days=1), periods=horizon_jours, freq="D")
    rows = []
    for i, d in enumerate(dates):
        val = max(0, last + trend * (i + 1))
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
    Hiver (11–2) : plus longue ; été (6–8) : plus courte. En jours.
    """
    if month in (11, 12, 1, 2):
        return base * 1.08
    if month in (6, 7, 8):
        return base * 0.92
    return base


# --- Occupation à partir des admissions (modèle stock + durée séjour) ---
def predict_occupation_from_admissions(
    occupation_df: pd.DataFrame,
    horizon_jours: int = 14,
    duree_sejour_moy: float = 6.0,
    use_best_admissions: bool = True,
    duree_sejour_saisonniere: bool = True,
) -> pd.DataFrame:
    """
    Prévision du taux d'occupation à partir des admissions prédites (modèle stock).
    Si duree_sejour_saisonniere=True (réf. Lequertier), la durée de séjour varie selon le mois.
    """
    occ = prepare_series(occupation_df, "occupation_lits")
    adm = prepare_series(occupation_df, "admissions_jour")
    occ = _ensure_daily_index(occ)
    adm = _ensure_daily_index(adm)

    if use_best_admissions:
        pred_adm = predict_admissions_best(adm, horizon_jours=horizon_jours)
    else:
        pred_adm = predict_moving_average(adm, horizon_jours=horizon_jours)

    occ_mean = occ.iloc[-28:].mean() if len(occ) >= 28 else occ.mean()
    adm_mean = adm.iloc[-28:].mean() if len(adm) >= 28 else adm.mean()
    ratio = occ_mean / adm_mean if adm_mean > 0 else 1

    pred = []
    for _, row in pred_adm.iterrows():
        adm_val = row["prediction"]
        d = row["date"]
        month = d.month if hasattr(d, "month") else pd.Timestamp(d).month
        duree = _duree_sejour_saisonniere(month, duree_sejour_moy) if duree_sejour_saisonniere else duree_sejour_moy
        occ_pred = occ_mean * 0.85 + adm_val * ratio * 0.15 * duree
        occ_low = row.get("prediction_low", adm_val - 50)
        occ_high = row.get("prediction_high", adm_val + 50)
        occ_pred_low = occ_mean * 0.85 + max(0, occ_low) * ratio * 0.15 * duree
        occ_pred_high = occ_mean * 0.85 + occ_high * ratio * 0.15 * duree
        pred.append({
            "date": d,
            "occupation_lits_pred": max(0, occ_pred),
            "occupation_lits_low": max(0, occ_pred_low),
            "occupation_lits_high": occ_pred_high,
            "admissions_pred": adm_val,
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
        occupation_df, horizon_jours=horizon_jours, duree_sejour_moy=duree_sejour_moy, use_best_admissions=True
    )
    from_adm["taux_occupation_pred"] = from_adm["occupation_lits_pred"] / capacite_lits
    from_adm["taux_occupation_low"] = from_adm["occupation_lits_low"] / capacite_lits
    from_adm["taux_occupation_high"] = from_adm["occupation_lits_high"] / capacite_lits
    return from_adm


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
        pred_df = predict_occupation_from_admissions(occupation_df, horizon_jours=horizon_jours)
        pred_df["taux_occupation_pred"] = pred_df["occupation_lits_pred"] / capacite_lits
        pred_df["taux_occupation_low"] = pred_df["taux_occupation_pred"] * 0.95
        pred_df["taux_occupation_high"] = pred_df["taux_occupation_pred"] * 1.05

    seuil_alerte = 0.85
    seuil_critique = 0.95
    pred_df["alerte"] = "normal"
    pred_df.loc[pred_df["taux_occupation_pred"] >= seuil_critique, "alerte"] = "critique"
    pred_df.loc[
        (pred_df["taux_occupation_pred"] >= seuil_alerte) & (pred_df["taux_occupation_pred"] < seuil_critique),
        "alerte",
    ] = "alerte"

    max_occ = pred_df["taux_occupation_pred"].max()
    max_high = pred_df["taux_occupation_high"].max() if "taux_occupation_high" in pred_df.columns else max_occ
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
        "taux_max_prevu": float(max_occ),
        "taux_max_high": float(max_high),
        "recommandation": reco,
        "seuils": {"alerte": seuil_alerte, "critique": seuil_critique},
    }


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
    series = _ensure_daily_index(series)
    if use_best:
        return predict_admissions_best(series, horizon_jours=horizon_jours)
    return predict_moving_average(series, horizon_jours=horizon_jours)
