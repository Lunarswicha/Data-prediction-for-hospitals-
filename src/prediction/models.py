"""
Modèles de prédiction — admissions, occupation, besoins (lits, personnel).

Moteur principal : Holt-Winters Triple Exponential Smoothing (statsmodels).
- Configuration apprise par validation (backtest) : trend (add / none) et damped_trend
  sont sélectionnés pour minimiser l'erreur (MAE) sur une fenêtre train/val (pas de réglage manuel).
- seasonal='add', seasonal_periods=7 (saisonnalité hebdomadaire apprise).
- Intervalles de confiance : écart-type des résidus sur les 30 derniers jours (Pred ± 1.96 * std_resid).
- Conversion Admissions → Occupation : modèle stock-flux (physiquement correct).
- Référence : Bouteloup (±10 %), Lequertier (DMS saisonnière).
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

from statsmodels.tsa.holtwinters import ExponentialSmoothing

try:
    from src.prediction.calendar_utils import (
        is_jour_ferie,
        is_veille_ferie,
        is_lendemain_ferie,
        is_vacances_scolaires_zone_c,
        temperature_synthetique_paris,
    )
except ImportError:
    def _no_calendar(d: Any) -> bool:
        return False
    def _temp_const(d: Any) -> float:
        return 15.0
    is_jour_ferie = is_veille_ferie = is_lendemain_ferie = is_vacances_scolaires_zone_c = _no_calendar
    temperature_synthetique_paris = _temp_const


# ---------------------------------------------------------------------------
# Préparation des séries (fréquence stricte pour ExponentialSmoothing)
# ---------------------------------------------------------------------------

def prepare_series(occupation_df: pd.DataFrame, col: str = "admissions_jour") -> pd.Series:
    """Série temporelle quotidienne avec index date."""
    df = occupation_df[["date", col]].drop_duplicates(subset=["date"]).sort_values("date")  # type: ignore[arg-type]
    return cast(pd.Series, df.set_index("date")[col])


def _ensure_daily_index(series: pd.Series) -> pd.Series:
    """
    Réindexe sur une plage de dates quotidienne continue (freq='D'),
    comble les trous par interpolation. Exigé par statsmodels ExponentialSmoothing.
    """
    if series.empty:
        return series
    dr = pd.date_range(series.index.min(), series.index.max(), freq="D")
    out = series.reindex(dr)
    out = out.interpolate(method="linear").ffill().bfill()
    out = out.asfreq("D")
    return out


# ---------------------------------------------------------------------------
# Holt-Winters Triple Exponential Smoothing (statsmodels — référence)
# ---------------------------------------------------------------------------

def predict_holt_winters_statsmodels(
    series: pd.Series,
    horizon_jours: int = 14,
    seasonal_periods: int = 7,
    resid_window_days: int = 30,
    trend: Optional[str] = None,
    damped_trend: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Prévision par Holt-Winters Triple Exponential Smoothing (statsmodels).
    Les paramètres trend et damped_trend sont estimés par validation (voir select_best_holt_winters_config)
    si non fournis ; sinon utilisation des valeurs par défaut (add, False).
    - trend : 'add', None (pas de tendance).
    - damped_trend : si trend='add', amortit la tendance sur l'horizon.
    - seasonal='add', seasonal_periods=7 : saisonnalité hebdo apprise.
    - initialization_method='estimated', fit(optimized=True) : apprentissage des paramètres sur les données.
    IC 95 % : Pred ± 1.96 * std(résidus sur les resid_window_days derniers jours).
    """
    series = _ensure_daily_index(series)
    if series.empty or len(series) < 2 * seasonal_periods:
        return None

    y = series.astype(float)
    if y.isna().any():
        y = y.ffill().bfill()
    if (y <= 0).any():
        y = y.clip(lower=1e-6)

    trend_val = trend if trend is not None else "add"
    damped = bool(damped_trend and trend_val == "add")

    try:
        model = ExponentialSmoothing(
            y,
            trend=trend_val,
            damped_trend=damped,
            seasonal="add",
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
            freq="D",
        )
        fit = model.fit(optimized=True, remove_bias=False)
    except Exception:
        return None

    forecast = fit.forecast(steps=horizon_jours)
    pred = np.asarray(forecast, dtype=float)
    pred = np.maximum(0, pred)

    # Intervalle de confiance : écart-type des résidus sur les N derniers jours
    resid = np.asarray(fit.resid).ravel()
    n_resid = min(resid_window_days, len(resid))
    std_resid = float(np.nanstd(resid[-n_resid:]) if n_resid > 0 else (pred.mean() * 0.10))
    if std_resid <= 0:
        std_resid = pred.mean() * 0.10

    ic_low = np.maximum(0, pred - 1.96 * std_resid)
    ic_high = pred + 1.96 * std_resid

    last_idx = cast(Any, series.index[-1])
    start_ts = pd.Timestamp(last_idx) + pd.Timedelta(days=1)
    dates = pd.date_range(start=start_ts, periods=horizon_jours, freq="D")

    return pd.DataFrame({
        "date": dates,
        "prediction": pred,
        "prediction_low": ic_low,
        "prediction_high": ic_high,
        "type": "admissions",
    })


def predict_holt_winters(
    series: pd.Series,
    horizon_jours: int = 14,
    seasonal_period: int = 7,
) -> Optional[pd.DataFrame]:
    """
    Alias public : Holt-Winters statsmodels (triple lissage exponentiel).
    seasonal_period conservé pour compatibilité API (utilisé comme seasonal_periods).
    """
    return predict_holt_winters_statsmodels(
        series,
        horizon_jours=horizon_jours,
        seasonal_periods=seasonal_period,
    )


def select_best_holt_winters_config(
    series: pd.Series,
    validation_days: int = 90,
    horizon_jours: Optional[int] = None,
    metric: str = "mae",
) -> Dict[str, Any]:
    """
    Sélection de la configuration Holt-Winters par validation (backtest).
    Entraînement sur [début, fin - validation_days], évaluation sur les validation_days derniers jours.
    Retourne la config (trend, damped_trend) qui minimise la métrique (mae ou maximise pct_within_10).
    Utilisé par predict_admissions_best pour apprendre la meilleure config à partir des données.
    """
    series = _ensure_daily_index(series)
    horizon = horizon_jours if horizon_jours is not None else min(validation_days, 30)
    if len(series) < validation_days + 2 * 7:
        return {"trend": "add", "damped_trend": False}
    train_end = len(series) - validation_days
    train = series.iloc[:train_end]
    test = series.iloc[train_end:]
    actual = test.values[:validation_days]
    if len(actual) < 7:
        return {"trend": "add", "damped_trend": False}

    configs: Tuple[Dict[str, Any], ...] = (
        {"trend": "add", "damped_trend": False},
        {"trend": "add", "damped_trend": True},
        {"trend": None, "damped_trend": False},
    )
    best_config = configs[0]
    best_score: float = float("inf") if metric == "mae" else float("-inf")

    for cfg in configs:
        try:
            pred_df = predict_holt_winters_statsmodels(
                train,
                horizon_jours=len(actual),
                trend=cfg.get("trend"),
                damped_trend=bool(cfg.get("damped_trend", False)),
            )
            if pred_df is None or pred_df.empty or len(pred_df) < len(actual):
                continue
            pred = pred_df["prediction"].values[: len(actual)]
            mae = float(np.mean(np.abs(pred - actual)))
            rel_err = np.where(actual > 0, np.abs(pred - actual) / np.maximum(actual, 1e-6), 0)
            pct_within_10 = float((rel_err <= 0.10).mean())
            if metric == "mae":
                score = mae
                if score < best_score:
                    best_score = score
                    best_config = cfg
            else:
                score = pct_within_10
                if score > best_score:
                    best_score = score
                    best_config = cfg
        except Exception:
            continue

    return dict(best_config)


# ---------------------------------------------------------------------------
# Baseline : moyenne glissante
# ---------------------------------------------------------------------------

def predict_moving_average(
    series: pd.Series,
    horizon_jours: int = 14,
    window: int = 28,
) -> pd.DataFrame:
    """Prévision par moyenne glissante (tendance = 0). Retourne date, prediction, prediction_low, prediction_high."""
    series = _ensure_daily_index(series)
    if len(series) < 7:
        last = float(series.mean())
        std = float(series.std()) if series.std() is not None else last * 0.1
    else:
        last = float(series.iloc[-window:].mean())
        std = float(series.iloc[-window:].std()) if series.iloc[-window:].std() is not None else last * 0.08
    start_ts = pd.Timestamp(cast(Any, series.index[-1])) + pd.Timedelta(days=1)
    dates = pd.date_range(start=start_ts, periods=horizon_jours, freq="D")
    pred = [max(0, last) for _ in range(horizon_jours)]
    return pd.DataFrame({
        "date": dates,
        "prediction": pred,
        "prediction_low": [max(0, p - 1.96 * std) for p in pred],
        "prediction_high": [p + 1.96 * std for p in pred],
        "type": "admissions",
    })


# ---------------------------------------------------------------------------
# Saisonnalité annuelle (hiver > été) — correction post-HW pour que le ML reflète la réalité
# ---------------------------------------------------------------------------

def _yearly_seasonal_factors(series: pd.Series) -> Dict[int, float]:
    """
    Facteurs mensuels (mois -> facteur) à partir de l'historique : moyenne du mois / moyenne globale.
    Permet d'ajuster les prévisions HW (saisonnalité 7 j seulement) pour que hiver > été.
    """
    if series.empty or len(series) < 60:
        return {}
    s = series.astype(float)
    idx = pd.DatetimeIndex(s.index) if not isinstance(s.index, pd.DatetimeIndex) else s.index
    months = pd.Series(idx).dt.month.values
    global_mean = float(s.mean())
    if global_mean <= 0:
        return {}
    factors: Dict[int, float] = {}
    for m in range(1, 13):
        mask = months == m
        if mask.sum() < 3:
            factors[m] = 1.0
        else:
            factors[m] = float(s.loc[mask].mean() / global_mean)
    return factors


def _apply_yearly_seasonality(pred_df: pd.DataFrame, series: pd.Series) -> pd.DataFrame:
    """
    Multiplie les prévisions (prediction, prediction_low, prediction_high) par le facteur
    saisonnier annuel du mois de chaque date. Ainsi les prévisions baissent en été et montent en hiver.
    """
    factors = _yearly_seasonal_factors(series)
    if not factors:
        return pred_df
    out = pred_df.copy()
    dates = pred_df["date"]
    if hasattr(dates, "dt"):
        months = dates.dt.month
    else:
        months = pd.to_datetime(dates).month
    for col in ("prediction", "prediction_low", "prediction_high"):
        if col not in out.columns:
            continue
        mult = np.array([factors.get(int(m), 1.0) for m in months])
        out[col] = np.maximum(0, out[col].values * mult)
    return out


# ---------------------------------------------------------------------------
# Choix du meilleur modèle (admissions) — HW statsmodels par défaut
# ---------------------------------------------------------------------------

def _holt_winters_tuned(series: pd.Series, horizon_jours: int, validation_days: int) -> Optional[pd.DataFrame]:
    """Holt-Winters avec config apprise par validation + correction saisonnalité annuelle (hiver > été)."""
    best_config = select_best_holt_winters_config(
        series, validation_days=validation_days, horizon_jours=horizon_jours, metric="mae"
    )
    result = predict_holt_winters_statsmodels(
        series,
        horizon_jours=horizon_jours,
        trend=best_config.get("trend"),
        damped_trend=bool(best_config.get("damped_trend", False)),
    )
    if result is not None and len(result) >= horizon_jours:
        result = _apply_yearly_seasonality(result, series)
    return result


def predict_admissions_best(
    series: pd.Series,
    horizon_jours: int = 14,
    prefer: str = "holt_winters",
) -> pd.DataFrame:
    """
    Prévision des admissions. Par défaut : Holt-Winters (statsmodels) avec configuration
    apprise par validation (train/val backtest) : trend (add / none) et damped_trend
    sont choisis pour minimiser l'erreur sur une fenêtre de validation.
    - prefer="holt_winters" (défaut) : HW avec config sélectionnée par les données.
    - prefer="best_by_backtest" : benchmark HW / Ridge / Boosting / SARIMA.
    - Fallback : moyenne glissante.
    """
    series = _ensure_daily_index(series)
    if series.empty or len(series) < 7:
        return predict_moving_average(series, horizon_jours=horizon_jours)

    validation_days = min(90, max(28, len(series) // 3))

    if prefer == "holt_winters" or prefer == "holt_winters_stable":
        result = _holt_winters_tuned(series, horizon_jours, validation_days)
        if result is not None and len(result) >= horizon_jours:
            return result.head(horizon_jours)

    if prefer == "best_by_backtest":
        best_name, _ = select_best_model_by_backtest(series, validation_days=validation_days)
        def _hw(s: pd.Series, horizon_jours: int = 14) -> Optional[pd.DataFrame]:
            return _holt_winters_tuned(s, horizon_jours, validation_days)
        model_map = {
            "holt_winters": _hw,
            "regression": predict_regression,
            "boosting": predict_boosting,
            "sarima": predict_sarima,
        }
        if best_name in model_map:
            try:
                result = model_map[best_name](series, horizon_jours=horizon_jours)
                if result is not None and not (hasattr(result, "empty") and result.empty) and len(result) >= horizon_jours:
                    return result.head(horizon_jours)
            except Exception:
                pass

    def _hw_best(s: pd.Series, horizon_jours: int = 14) -> Optional[pd.DataFrame]:
        return _holt_winters_tuned(s, horizon_jours, validation_days)
    order: Tuple[Tuple[str, Any], ...] = (
        ("holt_winters", _hw_best),
        ("regression", predict_regression),
        ("boosting", predict_boosting),
        ("sarima", predict_sarima),
    )
    for _name, fn in order:
        try:
            res = fn(series, horizon_jours=horizon_jours)
            if res is not None and not (hasattr(res, "empty") and res.empty) and len(res) >= horizon_jours:
                return res.head(horizon_jours)
        except Exception:
            continue

    return predict_moving_average(series, horizon_jours=horizon_jours)


# ---------------------------------------------------------------------------
# Régression : lags + calendrier (Bouteloup)
# ---------------------------------------------------------------------------

def _build_lag_calendar_features(
    series: pd.Series,
    horizon: int,
    use_calendar_ferie: bool = True,
    use_temperature: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DatetimeIndex]:
    """Construit X_train, y_train, X_pred, pred_dates (lags 1,7,14, mean 7-13, calendrier)."""
    series = _ensure_daily_index(series)
    df = pd.DataFrame({"y": series})
    idx = pd.DatetimeIndex(df.index) if not isinstance(df.index, pd.DatetimeIndex) else df.index
    for lag in [1, 7, 14]:
        if len(series) > lag:
            df["lag_%d" % lag] = df["y"].shift(lag)
    if len(series) >= 14:
        lag_7_13 = sum(df["y"].shift(k) for k in range(7, 14)) / 7.0
        df["lag_mean_7_13"] = lag_7_13
    df["jour_semaine"] = getattr(cast(Any, idx), "dayofweek", 0)
    df["mois"] = getattr(cast(Any, idx), "month", 1)
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
    y_train = cast(pd.Series, df["y"])
    start_ts = pd.Timestamp(cast(Any, series.index[-1])) + pd.Timedelta(days=1)
    pred_dates = pd.date_range(start=start_ts, periods=horizon, freq="D")
    X_pred_list = []
    for d in pred_dates:
        row = {}
        for lag in [1, 7, 14]:
            key = "lag_%d" % lag
            if key in feature_cols:
                row[key] = series.get(d - pd.Timedelta(days=lag), np.nan)
        if "lag_mean_7_13" in feature_cols:
            vals = np.array([series.get(d - pd.Timedelta(days=k), np.nan) for k in range(7, 14)], dtype=float)
            row["lag_mean_7_13"] = float(np.nanmean(vals))
        row["jour_semaine"] = d.dayofweek
        row["mois"] = d.month
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
        if c in X_train.columns and bool(pd.Series(X_pred[c]).isna().any()):
            X_pred[c] = X_pred[c].fillna(X_train[c].mean())
    if bool(X_pred.isna().any(axis=None)):
        X_pred = X_pred.fillna(series.iloc[-28:].mean())
    return cast(Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DatetimeIndex], (X_train, y_train, X_pred, pred_dates))


def predict_regression(
    series: pd.Series,
    horizon_jours: int = 14,
    use_splines: bool = True,
) -> Optional[pd.DataFrame]:
    """Prévision Ridge sur lags + calendrier (option splines)."""
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler, SplineTransformer
    except ImportError:
        return None
    out = _build_lag_calendar_features(series, horizon_jours)
    X_train, y_train, X_pred, pred_dates = out
    if X_train.empty or X_pred.empty:
        return None
    spl_cols = [c for c in ["jour_semaine", "temperature"] if c in X_train.columns]
    if use_splines and spl_cols:
        other = [c for c in X_train.columns if c not in spl_cols]
        Xt_other = X_train[other].copy()
        Xp_other = X_pred[other].copy()
        for c in other:
            col = Xp_other[c]
            if isinstance(col, pd.Series) and col.isna().any():
                Xp_other[c] = col.fillna(Xt_other[c].mean())
        spline = SplineTransformer(n_knots=5, degree=3)
        Xt_spl = np.asarray(spline.fit_transform(X_train[spl_cols]), dtype=np.float64)
        Xp_spl = np.asarray(spline.transform(X_pred[spl_cols]), dtype=np.float64)
        Xt = np.hstack((np.asarray(Xt_other.values, dtype=np.float64), Xt_spl))
        Xp = np.hstack((np.asarray(Xp_other.values, dtype=np.float64), Xp_spl))
    else:
        Xt = X_train.values
        Xp = X_pred.values
    scaler = StandardScaler()
    Xt = scaler.fit_transform(Xt)
    Xp = scaler.transform(Xp)
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(Xt, y_train)
    pred = np.maximum(0, model.predict(Xp))
    resid = y_train - model.predict(Xt)
    std = float(resid.std() or np.abs(pred).mean() * 0.1)
    return pd.DataFrame({
        "date": pred_dates,
        "prediction": pred,
        "prediction_low": np.maximum(0, pred - 1.96 * std),
        "prediction_high": pred + 1.96 * std,
        "type": "admissions",
    })


# ---------------------------------------------------------------------------
# Boosting (XGBoost / GBM)
# ---------------------------------------------------------------------------

def predict_boosting(
    series: pd.Series,
    horizon_jours: int = 14,
    use_splines: bool = True,
) -> Optional[pd.DataFrame]:
    """Prévision par boosting (mêmes features que Ridge)."""
    try:
        import xgboost as xgb  # type: ignore[import-untyped]
        _use_xgb = True
    except ImportError:
        xgb = None  # type: ignore[assignment]
        _use_xgb = False
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler, SplineTransformer
    except ImportError:
        return None
    out = _build_lag_calendar_features(series, horizon_jours)
    X_train, y_train, X_pred, pred_dates = out
    if X_train.empty or X_pred.empty:
        return None
    spl_cols = [c for c in ["jour_semaine", "temperature"] if c in X_train.columns]
    if use_splines and spl_cols:
        other = [c for c in X_train.columns if c not in spl_cols]
        Xt_other = X_train[other].copy()
        Xp_other = X_pred[other].copy()
        for c in other:
            col = Xp_other[c]
            if isinstance(col, pd.Series) and col.isna().any():
                Xp_other[c] = col.fillna(Xt_other[c].mean())
        spline = SplineTransformer(n_knots=5, degree=3)
        Xt_spl = np.asarray(spline.fit_transform(X_train[spl_cols]), dtype=np.float64)
        Xp_spl = np.asarray(spline.transform(X_pred[spl_cols]), dtype=np.float64)
        Xt = np.hstack((np.asarray(Xt_other.values, dtype=np.float64), Xt_spl))
        Xp = np.hstack((np.asarray(Xp_other.values, dtype=np.float64), Xp_spl))
    else:
        Xt = X_train.values
        Xp = X_pred.values
    scaler = StandardScaler()
    Xt = scaler.fit_transform(Xt)
    Xp = scaler.transform(Xp)
    if _use_xgb and xgb is not None:
        model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, objective="reg:squarederror")
    else:
        model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(Xt, y_train)
    pred = np.maximum(0, model.predict(Xp))
    resid = y_train - model.predict(Xt)
    std = float(resid.std() or np.abs(pred).mean() * 0.1)
    return pd.DataFrame({
        "date": pred_dates,
        "prediction": pred,
        "prediction_low": np.maximum(0, pred - 1.96 * std),
        "prediction_high": pred + 1.96 * std,
        "type": "admissions",
    })


def evaluate_boosting_model(
    series: pd.Series,
    validation_days: int = 90,
) -> Dict[str, Any]:
    """Évalue le modèle boosting vs predict_admissions_best (HW)."""
    series = _ensure_daily_index(series)
    if len(series) < validation_days + 60:
        return {"mae": None, "rmse": None, "pct_within_10": None, "mean_error": None, "message": "Série trop courte.", "boosting_vs_best": None}
    train = series.iloc[:-validation_days]
    test = series.iloc[-validation_days:]
    horizon = len(test)
    pred_boosting = predict_boosting(train, horizon_jours=horizon)
    pred_best = predict_admissions_best(train, horizon_jours=horizon)
    if pred_boosting is None or pred_boosting.empty or len(pred_boosting) < horizon:
        return {"mae": None, "rmse": None, "pct_within_10": None, "mean_error": None, "message": "Échec boosting.", "boosting_vs_best": None}
    if pred_best is None or pred_best.empty or len(pred_best) < horizon:
        pred_best = predict_moving_average(train, horizon_jours=horizon)
    actual = test.values[:horizon]
    p_boost = pred_boosting["prediction"].values[:horizon]
    p_best = pred_best["prediction"].values[:horizon]
    mae = float(np.mean(np.abs(p_boost - actual)))
    rmse = float(np.sqrt(np.mean((p_boost - actual) ** 2)))
    rel_err = np.where(actual > 0, np.abs(p_boost - actual) / np.maximum(actual, 1e-6), 0)
    pct_within_10 = float((rel_err <= 0.10).mean())
    mae_best = float(np.mean(np.abs(p_best - actual)))
    rmse_best = float(np.sqrt(np.mean((p_best - actual) ** 2)))
    return {
        "mae": mae,
        "rmse": rmse,
        "pct_within_10": pct_within_10,
        "mean_error": float(np.mean(p_boost - actual)),
        "n_days": horizon,
        "message": None,
        "boosting_vs_best": {
            "mae_boosting": mae,
            "mae_best": mae_best,
            "rmse_boosting": rmse,
            "rmse_best": rmse_best,
            "meilleur_mae": "boosting" if mae <= mae_best else "best",
            "meilleur_rmse": "boosting" if rmse <= rmse_best else "best",
        },
    }


# ---------------------------------------------------------------------------
# SARIMA (optionnel)
# ---------------------------------------------------------------------------

def predict_sarima(
    series: pd.Series,
    horizon_jours: int = 14,
    order: Tuple[int, int, int] = (1, 0, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 7),
) -> Optional[pd.DataFrame]:
    """Prévision SARIMA avec saisonnalité 7 j."""
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        return None
    series = _ensure_daily_index(series)
    if len(series) < 4 * seasonal_order[3]:
        return None
    y = series.clip(lower=1e-6)
    try:
        model = SARIMAX(y, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False, maxiter=100)
    except Exception:
        return None
    forecast = getattr(fit, "get_forecast")(steps=horizon_jours)
    try:
        sf = forecast.summary_frame(alpha=0.05)
    except Exception:
        sf = forecast.summary_frame()
    pred = np.asarray(sf["mean"]).ravel()
    if "mean_ci_lower" in sf.columns and "mean_ci_upper" in sf.columns:
        low = np.asarray(sf["mean_ci_lower"]).ravel()
        high = np.asarray(sf["mean_ci_upper"]).ravel()
    else:
        se = np.asarray(sf["mean_se"]).ravel() if "mean_se" in sf.columns else np.full_like(pred, pred.mean() * 0.1)
        low = np.maximum(0, pred - 1.96 * se)
        high = pred + 1.96 * se
    start_ts = pd.Timestamp(cast(Any, series.index[-1])) + pd.Timedelta(days=1)
    dates = pd.date_range(start=start_ts, periods=horizon_jours, freq="D")
    return pd.DataFrame({
        "date": dates,
        "prediction": np.maximum(0, pred),
        "prediction_low": np.maximum(0, low),
        "prediction_high": high,
        "type": "admissions",
    })


# ---------------------------------------------------------------------------
# Benchmark : sélection du meilleur modèle par backtest
# ---------------------------------------------------------------------------

def select_best_model_by_backtest(
    series: pd.Series,
    validation_days: int = 90,
    metric_primary: str = "pct_within_10",
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """Backteste HW (statsmodels), Ridge, Boosting, SARIMA. Retourne le meilleur et les métriques."""
    series = _ensure_daily_index(series)
    if len(series) < validation_days + 60:
        return "holt_winters", {}
    train_end = len(series) - validation_days
    train = series.iloc[:train_end]
    test = series.iloc[train_end:]
    horizon = len(test)
    actual = test.values
    def _hw_fn(s: pd.Series, horizon_jours: int) -> Optional[pd.DataFrame]:
        return predict_holt_winters_statsmodels(s, horizon_jours=horizon_jours)

    models_to_test: Tuple[Tuple[str, Any], ...] = (
        ("holt_winters", _hw_fn),
        ("regression", predict_regression),
        ("boosting", predict_boosting),
        ("sarima", predict_sarima),
    )
    results: Dict[str, Dict[str, float]] = {}
    for name, fn in models_to_test:
        try:
            pred_df = fn(train, horizon_jours=horizon)
            if pred_df is None or pred_df.empty or len(pred_df) < horizon:
                continue
            pred = pred_df["prediction"].values[:horizon]
            mae = float(np.mean(np.abs(pred - actual)))
            rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
            rel_err = np.where(actual > 0, np.abs(pred - actual) / np.maximum(actual, 1e-6), 0)
            pct_within_10 = float((rel_err <= 0.10).mean())
            results[name] = {"mae": mae, "rmse": rmse, "pct_within_10": pct_within_10}
        except Exception:
            continue
    if not results:
        return "holt_winters", {}
    def score(n: str) -> Tuple[float, float]:
        r = results[n]
        return (-r["pct_within_10"], r["mae"])
    best_name = min(results.keys(), key=score)
    return best_name, results


# ---------------------------------------------------------------------------
# Occupation : prédiction directe (série occupation_lits)
# ---------------------------------------------------------------------------

def predict_occupation_direct(
    occupation_df: pd.DataFrame,
    horizon_jours: int = 14,
    prefer: str = "auto",
) -> pd.DataFrame:
    """Prévision directe de l'occupation (HW statsmodels ou Ridge, puis MA)."""
    occ = prepare_series(occupation_df, "occupation_lits")
    occ = _ensure_daily_index(occ)
    if occ.empty:
        return pd.DataFrame()
    if prefer != "regression":
        result = predict_holt_winters_statsmodels(occ, horizon_jours=horizon_jours)
        if result is not None and len(result) == horizon_jours:
            result = result.rename(columns={
                "prediction": "occupation_lits_pred",
                "prediction_low": "occupation_lits_low",
                "prediction_high": "occupation_lits_high",
            })
            result["admissions_pred"] = np.nan
            return cast(pd.DataFrame, result[["date", "occupation_lits_pred", "occupation_lits_low", "occupation_lits_high", "admissions_pred"]])
    if prefer != "holt_winters":
        result = predict_regression(occ, horizon_jours=horizon_jours)
        if result is not None and len(result) == horizon_jours:
            result = result.rename(columns={
                "prediction": "occupation_lits_pred",
                "prediction_low": "occupation_lits_low",
                "prediction_high": "occupation_lits_high",
            })
            result["admissions_pred"] = np.nan
            return cast(pd.DataFrame, result[["date", "occupation_lits_pred", "occupation_lits_low", "occupation_lits_high", "admissions_pred"]])
    window = 28
    last = float(occ.iloc[-window:].mean())
    std = float(occ.iloc[-window:].std()) if occ.iloc[-window:].std() is not None else last * 0.05
    start_ts = pd.Timestamp(cast(Any, occ.index[-1])) + pd.Timedelta(days=1)
    dates = pd.date_range(start=start_ts, periods=horizon_jours, freq="D")
    return pd.DataFrame({
        "date": dates,
        "occupation_lits_pred": [max(0, last)] * horizon_jours,
        "occupation_lits_low": [max(0, last - 1.96 * std)] * horizon_jours,
        "occupation_lits_high": [last + 1.96 * std] * horizon_jours,
        "admissions_pred": np.nan,
    })


# ---------------------------------------------------------------------------
# Durée de séjour saisonnière (Lequertier) et modèle stock Admissions → Occupation
# ---------------------------------------------------------------------------

def _duree_sejour_saisonniere(month: int, base: float = 6.0) -> float:
    """DMS selon le mois (variation sinusoïdale ±4 %, max hiver)."""
    angle = 2 * math.pi * (month - 1) / 12.0
    variation = 0.04 * math.cos(angle)
    return base * (1.0 + variation)


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
    Prévision de l'occupation à partir des admissions (modèle stock-flux).
    Stock(t) = Stock(t-1) + Entrées(t) - Sorties(t), Sorties(t) = Stock(t-1) / DMS(t).
    """
    if pred_admissions_df is None and pred_adm is not None:
        pred_admissions_df = pred_adm
    occ = prepare_series(occupation_df, "occupation_lits")
    adm = prepare_series(occupation_df, "admissions_jour")
    occ = _ensure_daily_index(occ)
    adm = _ensure_daily_index(adm)

    if duree_sejour_moy is None:
        occ_mean = float(occ.iloc[-90:].mean()) if len(occ) >= 90 else float(occ.mean())
        adm_mean = float(adm.iloc[-90:].mean()) if len(adm) >= 90 else float(adm.mean())
        duree_sejour_moy = occ_mean / adm_mean if adm_mean > 0 else 4.0
        duree_sejour_moy = max(2.0, min(10.0, float(duree_sejour_moy)))
    dms: float = duree_sejour_moy

    if pred_admissions_df is not None and len(pred_admissions_df) >= horizon_jours:
        pred_adm = pred_admissions_df.head(horizon_jours).copy()
        if "prediction" not in pred_adm.columns and len(pred_adm.columns) > 1:
            col1 = str(pred_adm.columns[1])
            pred_adm = pred_adm.rename(columns={col1: "prediction"})
    else:
        pred_adm = predict_admissions_best(adm, horizon_jours=horizon_jours) if use_best_admissions else predict_moving_average(adm, horizon_jours=horizon_jours)

    stock_actuel_pred = float(occ.iloc[-1])
    stock_actuel_low = stock_actuel_pred * 0.97
    stock_actuel_high = stock_actuel_pred * 1.03
    alpha_inertie = 0.5

    pred = []
    for _, row in pred_adm.iterrows():
        entrees_pred = float(row.get("prediction", 0) or 0)
        entrees_low = float(row.get("prediction_low", entrees_pred * 0.92) or 0)
        entrees_high = float(row.get("prediction_high", entrees_pred * 1.08) or 0)
        d = row["date"]
        d_val = d.iloc[0] if isinstance(d, pd.Series) else d
        d_ts = pd.Timestamp(cast(Any, d_val)) if not isinstance(d_val, pd.Timestamp) else d_val
        month = int(d_ts.month)
        duree = _duree_sejour_saisonniere(month, dms) if duree_sejour_saisonniere else dms
        sorties_pred = stock_actuel_pred / duree
        sorties_low = stock_actuel_low / duree
        sorties_high = stock_actuel_high / duree
        stock_brut_pred = stock_actuel_pred + entrees_pred - sorties_pred
        stock_brut_low = stock_actuel_low + entrees_low - sorties_low
        stock_brut_high = stock_actuel_high + entrees_high - sorties_high
        stock_actuel_pred = min(capacite_lits, max(0, (1 - alpha_inertie) * stock_actuel_pred + alpha_inertie * stock_brut_pred))
        stock_actuel_low = min(capacite_lits, max(0, (1 - alpha_inertie) * stock_actuel_low + alpha_inertie * stock_brut_low))
        stock_actuel_high = min(capacite_lits * 0.95, max(0, (1 - alpha_inertie) * stock_actuel_high + alpha_inertie * stock_brut_high))
        pred.append({
            "date": d_ts,
            "occupation_lits_pred": stock_actuel_pred,
            "occupation_lits_low": stock_actuel_low,
            "occupation_lits_high": stock_actuel_high,
            "admissions_pred": entrees_pred,
        })
    return pd.DataFrame(pred)


# ---------------------------------------------------------------------------
# Choix prédiction occupation et besoins
# ---------------------------------------------------------------------------

def predict_occupation_best(
    occupation_df: pd.DataFrame,
    horizon_jours: int = 14,
    capacite_lits: int = 1800,
    duree_sejour_moy: float = 6.0,
) -> pd.DataFrame:
    """Prévision occupation : directe (HW) si dispo, sinon via admissions (stock-flux)."""
    direct = predict_occupation_direct(occupation_df, horizon_jours=horizon_jours)
    if len(direct) > 0 and bool((direct["occupation_lits_pred"].notna()).all()):
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


def _besoins_from_pred_df(pred_df: pd.DataFrame, capacite_lits: int) -> Dict[str, Any]:
    """À partir des prévisions occupation : taux, alerte, recommandation."""
    if pred_df.empty:
        return {"previsions": pred_df, "taux_max_prevu": 0.0, "taux_max_high": 0.0, "recommandation": "Données insuffisantes.", "seuils": {"alerte": 0.85, "critique": 0.95}}
    p = pred_df.copy()
    if "taux_occupation_pred" not in p.columns:
        occ_pred = p["occupation_lits_pred"]
        p["taux_occupation_pred"] = occ_pred / capacite_lits
        if "occupation_lits_low" in p.columns:
            p["taux_occupation_low"] = p["occupation_lits_low"] / capacite_lits
        else:
            p["taux_occupation_low"] = occ_pred * 0.95 / capacite_lits
        if "occupation_lits_high" in p.columns:
            p["taux_occupation_high"] = p["occupation_lits_high"] / capacite_lits
        else:
            p["taux_occupation_high"] = occ_pred * 1.05 / capacite_lits
    seuil_alerte, seuil_critique = 0.85, 0.95
    p["alerte"] = "normal"
    p.loc[p["taux_occupation_pred"] >= seuil_critique, "alerte"] = "critique"
    p.loc[(p["taux_occupation_pred"] >= seuil_alerte) & (p["taux_occupation_pred"] < seuil_critique), "alerte"] = "alerte"
    max_occ = float(p["taux_occupation_pred"].max())
    max_high = float(p["taux_occupation_high"].max()) if "taux_occupation_high" in p.columns else max_occ
    if max_occ >= seuil_critique:
        reco = "Renforcer les effectifs et reporter les interventions non urgentes."
    elif max_occ >= seuil_alerte:
        reco = "Surveiller les effectifs et préparer une montée en charge."
    elif max_high >= seuil_alerte:
        reco = "Vigilance : la borne haute approche le seuil d'alerte."
    else:
        reco = "Capacité dans la norme ; maintenir la vigilance."
    return {"previsions": p, "taux_max_prevu": max_occ, "taux_max_high": max_high, "recommandation": reco, "seuils": {"alerte": seuil_alerte, "critique": seuil_critique}}


def build_besoins_from_occupation_pred(pred_df: pd.DataFrame, capacite_lits: int = 1800) -> Dict[str, Any]:
    """API publique : dict besoins à partir d'un DataFrame de prévisions occupation."""
    return _besoins_from_pred_df(pred_df, capacite_lits)


def predict_besoins(
    occupation_df: pd.DataFrame,
    capacite_lits: int = 1800,
    horizon_jours: int = 14,
    duree_sejour_moy: float = 6.0,
) -> Dict[str, Any]:
    """Prévision des besoins (lits, alerte) avec modèle principal (HW → occupation)."""
    pred_df = predict_occupation_best(occupation_df, horizon_jours=horizon_jours, capacite_lits=capacite_lits, duree_sejour_moy=duree_sejour_moy)
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
    """Prévision des besoins en forçant un modèle (auto | holt_winters | ridge | sarima | ma | boosting | direct_hw | direct_ridge)."""
    occ = prepare_series(occupation_df, "occupation_lits")
    adm = prepare_series(occupation_df, "admissions_jour")
    occ = _ensure_daily_index(occ)
    adm = _ensure_daily_index(adm)

    if model_choice == "auto":
        return predict_besoins(occupation_df, capacite_lits=capacite_lits, horizon_jours=horizon_jours, duree_sejour_moy=duree_sejour_moy)
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

    pred_adm: Optional[pd.DataFrame] = None
    if model_choice == "holt_winters":
        pred_adm = predict_holt_winters_statsmodels(adm, horizon_jours=horizon_jours)
    elif model_choice == "ridge":
        pred_adm = predict_regression(adm, horizon_jours=horizon_jours)
    elif model_choice == "sarima":
        pred_adm = predict_sarima(adm, horizon_jours=horizon_jours)
    elif model_choice == "ma":
        pred_adm = predict_moving_average(adm, horizon_jours=horizon_jours)
    elif model_choice == "boosting":
        pred_adm = predict_boosting(adm, horizon_jours=horizon_jours)

    if pred_adm is None or pred_adm.empty or len(pred_adm) < horizon_jours:
        return predict_besoins(occupation_df, capacite_lits=capacite_lits, horizon_jours=horizon_jours, duree_sejour_moy=duree_sejour_moy)
    pred_df = predict_occupation_from_admissions(
        occupation_df, horizon_jours=horizon_jours, duree_sejour_moy=duree_sejour_moy, pred_admissions_df=pred_adm, capacite_lits=capacite_lits
    )
    return _besoins_from_pred_df(pred_df, capacite_lits)


# ---------------------------------------------------------------------------
# Validation ±10 % (Bouteloup) et backtest
# ---------------------------------------------------------------------------

def evaluate_forecast_pct_within_10(
    series: pd.Series,
    validation_days: int = 90,
    use_best: bool = True,
) -> Dict[str, Any]:
    """% de jours à ±10 % sur la période de validation (réf. Bouteloup)."""
    series = _ensure_daily_index(series)
    if len(series) < validation_days + 60:
        return {"pct_within_10": None, "mean_error": None, "n_days": 0, "message": "Série trop courte"}
    train_end = len(series) - validation_days
    train = series.iloc[:train_end]
    test = series.iloc[train_end:]
    horizon = len(test)
    pred_df = predict_admissions_best(train, horizon_jours=horizon) if use_best else predict_moving_average(train, horizon_jours=horizon)
    if pred_df is None or pred_df.empty or len(pred_df) < horizon:
        return {"pct_within_10": None, "mean_error": None, "n_days": 0, "message": "Échec prédiction"}
    pred = pred_df["prediction"].values[:horizon]
    actual = test.values[:horizon]
    rel_err = np.where(actual > 0, np.abs(pred - actual) / np.maximum(actual, 1e-6), 0)
    return {
        "pct_within_10": float((rel_err <= 0.10).mean()),
        "mean_error": float(np.mean(pred - actual)),
        "n_days": horizon,
        "pct_surestimation": float((pred > actual * 1.10).mean()),
        "pct_sous_estimation": float((pred < actual * 0.90).mean()),
    }


def run_backtest_admissions(
    series: pd.Series,
    validation_days: int = 90,
    use_best: bool = True,
) -> Dict[str, Any]:
    """Backtest : train sur le passé, prédit la période de validation ; retourne DataFrame et métriques."""
    series = _ensure_daily_index(series)
    if len(series) < validation_days + 60:
        return {"backtest_df": pd.DataFrame(), "metrics": None, "message": "Série trop courte."}
    train_end = len(series) - validation_days
    train = series.iloc[:train_end]
    test = series.iloc[train_end:]
    horizon = len(test)
    pred_df = predict_admissions_best(train, horizon_jours=horizon) if use_best else predict_moving_average(train, horizon_jours=horizon)
    if pred_df is None or pred_df.empty or len(pred_df) < horizon:
        return {"backtest_df": pd.DataFrame(), "metrics": None, "message": "Échec prédiction."}
    actual = test.values[:horizon]
    pred = pred_df["prediction"].values[:horizon]
    dates = pd.to_datetime(pred_df["date"].values[:horizon])
    backtest_df = pd.DataFrame({"date": dates, "observé": actual, "prévu": pred})
    if "prediction_low" in pred_df.columns and "prediction_high" in pred_df.columns:
        backtest_df["prévu_basse"] = pred_df["prediction_low"].values[:horizon]
        backtest_df["prévu_haute"] = pred_df["prediction_high"].values[:horizon]
    rel_err = np.where(actual > 0, np.abs(pred - actual) / np.maximum(actual, 1e-6), 0)
    metrics = {
        "pct_within_10": float((rel_err <= 0.10).mean()),
        "mean_error": float(np.mean(pred - actual)),
        "mae": float(np.mean(np.abs(pred - actual))),
        "rmse": float(np.sqrt(np.mean((pred - actual) ** 2))),
        "n_days": horizon,
        "pct_surestimation": float((pred > actual * 1.10).mean()),
        "pct_sous_estimation": float((pred < actual * 0.90).mean()),
    }
    return {"backtest_df": backtest_df, "metrics": metrics, "message": None}


# ---------------------------------------------------------------------------
# Infos "ensemble" (dashboard : un seul modèle = HW statsmodels)
# ---------------------------------------------------------------------------

def get_ensemble_info(series: pd.Series) -> Dict[str, Any]:
    """Retourne les infos pour l'affichage dashboard (modèle = Holt-Winters, config apprise par validation)."""
    return {
        "top3_names": ["Holt-Winters (config apprise par validation)"],
        "weights": {"Holt-Winters (statsmodels)": 100.0},
        "all_performances": {},
        "validation_days": 28,
    }


# ---------------------------------------------------------------------------
# Prévision admissions (API agrégée)
# ---------------------------------------------------------------------------

def fit_and_predict_admissions(
    admissions_df: pd.DataFrame,
    service: Optional[str] = None,
    horizon_jours: int = 14,
    use_best: bool = True,
) -> pd.DataFrame:
    """Prévision des admissions (global ou par service). use_best=True → HW statsmodels."""
    if service:
        series = admissions_df[admissions_df["service"] == service].groupby("date")["admissions"].sum()
    else:
        series = admissions_df.groupby("date")["admissions"].sum()
    series = _ensure_daily_index(cast(pd.Series, series))
    if use_best:
        return predict_admissions_best(series, horizon_jours=horizon_jours)
    return predict_moving_average(series, horizon_jours=horizon_jours)
