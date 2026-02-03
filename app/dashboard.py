"""
Tableau de bord interactif — MVP Pitié-Salpêtrière.
Exploration des flux, prévisions, simulation de scénarios, recommandations.
Lancer : streamlit run app/dashboard.py (depuis la racine du projet)
"""

import sys
from pathlib import Path

# Racine du projet
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data.load import load_admissions, load_occupation
from src.prediction.models import (
    predict_besoins,
    evaluate_forecast_pct_within_10,
    evaluate_boosting_model,
    predict_boosting,
    prepare_series,
)
from src.simulation.scenarios import run_scenario, SCENARIOS
from src.recommendations.engine import (
    recommendations_from_forecast,
    recommendations_from_scenario,
    format_recommendations_for_dashboard,
)

st.set_page_config(
    page_title="MVP Pitié-Salpêtrière — Prévision des besoins",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 MVP Pitié-Salpêtrière")
st.caption("Simulation et prévision des besoins hospitaliers — Données fictives")

try:
    admissions_df = load_admissions(ROOT / "data")
    occupation_df = load_occupation(ROOT / "data")
except FileNotFoundError as e:
    st.error(
        "Données non trouvées. Exécuter d'abord : `python -m src.data.generate` depuis la racine du projet."
    )
    st.stop()

# Bornes des données pour les filtres
date_min_data = occupation_df["date"].min().date() if hasattr(occupation_df["date"].min(), "date") else pd.Timestamp(occupation_df["date"].min()).date()
date_max_data = occupation_df["date"].max().date() if hasattr(occupation_df["date"].max(), "date") else pd.Timestamp(occupation_df["date"].max()).date()

# Sidebar
st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Section",
    [
        "Flux & historique",
        "Prévisions",
        "Simulation de scénarios",
        "Modèle Boosting (apprentissage)",
        "Recommandations",
    ],
)

# --- Flux & historique ---
if section == "Flux & historique":
    st.header("Flux hospitaliers et historique")

    # Filtres par période (jour / mois / année)
    st.subheader("Filtrer la période")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        default_debut = (pd.Timestamp(date_max_data) - pd.DateOffset(months=12)).date()
        if default_debut < date_min_data:
            default_debut = date_min_data
        date_debut_hist = st.date_input(
            "Du (date)",
            value=default_debut,
            min_value=date_min_data,
            max_value=date_max_data,
            key="hist_debut",
        )
    with col_f2:
        date_fin_hist = st.date_input(
            "Au (date)",
            value=date_max_data,
            min_value=date_min_data,
            max_value=date_max_data,
            key="hist_fin",
        )
    with col_f3:
        st.caption("Réduire la période pour des courbes plus lisibles.")

    if date_debut_hist > date_fin_hist:
        st.warning("La date de début doit être antérieure à la date de fin. Affichage de la période complète.")
        date_debut_hist, date_fin_hist = date_min_data, date_max_data

    ts_debut = pd.Timestamp(date_debut_hist)
    ts_fin = pd.Timestamp(date_fin_hist)
    mask_occ = (occupation_df["date"] >= ts_debut) & (occupation_df["date"] <= ts_fin)
    mask_adm = (admissions_df["date"] >= ts_debut) & (admissions_df["date"] <= ts_fin)
    occ_filtree = occupation_df.loc[mask_occ].copy()
    adm_filtree = admissions_df.loc[mask_adm].copy()

    st.subheader("Visibilité des graphiques")
    services_dispo = sorted(adm_filtree["service"].unique().tolist()) if not adm_filtree.empty else []
    col_v1, col_v2, col_v3 = st.columns(3)
    with col_v1:
        services_choisis = st.multiselect(
            "Services à afficher (admissions)",
            options=services_dispo,
            default=services_dispo[:5] if len(services_dispo) > 5 else services_dispo,
            key="hist_services",
            help="Sélectionnez 2 à 5 services pour des courbes plus lisibles. Vide = tous.",
        )
        if not services_choisis and services_dispo:
            services_choisis = services_dispo
    with col_v2:
        lissage_7j = st.checkbox(
            "Lisser les courbes (moyenne glissante 7 j)",
            value=False,
            key="hist_lissage",
            help="Réduit le bruit et améliore la lisibilité des tendances.",
        )
    with col_v3:
        agg_periode = st.radio(
            "Agrégation des points",
            options=["Quotidien", "Hebdomadaire"],
            index=0,
            key="hist_agg",
            help="Hebdomadaire : moins de points, courbes moins condensées.",
        )
        freq_agg = "W-MON" if agg_periode == "Hebdomadaire" else None

    st.caption(
        f"Période : **{date_debut_hist.strftime('%d/%m/%Y')}** → **{date_fin_hist.strftime('%d/%m/%Y')}** "
        f"({len(occ_filtree)} jours) · Services affichés : {len(services_choisis) if services_choisis else len(services_dispo)}"
    )

    adm_agg = (
        adm_filtree.groupby(["date", "service"])["admissions"]
        .sum()
        .reset_index()
    )
    if not adm_agg.empty and services_choisis:
        adm_agg = adm_agg[adm_agg["service"].isin(services_choisis)]
    if not adm_agg.empty and freq_agg:
        adm_agg = adm_agg.copy()
        adm_agg["date"] = pd.to_datetime(adm_agg["date"])
        adm_agg = (
            adm_agg.groupby([pd.Grouper(key="date", freq=freq_agg), "service"])["admissions"]
            .sum()
            .reset_index()
        )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Admissions par service")
        if adm_agg.empty:
            st.info("Aucune donnée sur cette période (ou aucun service sélectionné).")
        else:
            if lissage_7j and agg_periode == "Quotidien":
                adm_smooth = (
                    adm_agg.sort_values(["service", "date"])
                    .groupby("service")["admissions"]
                    .transform(lambda s: s.rolling(7, min_periods=1).mean())
                )
                adm_agg = adm_agg.copy()
                adm_agg["admissions"] = adm_smooth
            fig_adm = px.line(
                adm_agg,
                x="date",
                y="admissions",
                color="service",
                title="Évolution des admissions par service",
            )
            fig_adm.update_layout(height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig_adm.update_xaxes(tickformat="%d/%m/%Y")
            st.plotly_chart(fig_adm, use_container_width=True)

    with col2:
        st.subheader("Occupation des lits (taux)")
        occ_plot = occ_filtree.copy()
        occ_plot["taux_pct"] = occ_plot["taux_occupation"] * 100
        if freq_agg:
            occ_plot["date"] = pd.to_datetime(occ_plot["date"])
            occ_plot = occ_plot.set_index("date").resample(freq_agg).mean().reset_index()
            occ_plot["taux_pct"] = occ_plot["taux_occupation"] * 100
        if lissage_7j and not occ_plot.empty:
            occ_plot["taux_pct"] = occ_plot["taux_pct"].rolling(7, min_periods=1).mean()
        if occ_plot.empty:
            st.info("Aucune donnée sur cette période.")
        else:
            fig_occ = px.line(
                occ_plot,
                x="date",
                y="taux_pct",
                title="Taux d'occupation quotidien",
            )
            fig_occ.add_hline(y=85, line_dash="dash", line_color="orange")
            fig_occ.add_hline(y=95, line_dash="dash", line_color="red")
            fig_occ.update_layout(height=400)
            fig_occ.update_xaxes(tickformat="%d/%m/%Y")
            st.plotly_chart(fig_occ, use_container_width=True)

    st.subheader("Répartition des admissions par service (sur la période filtrée)")
    if adm_filtree.empty:
        part = admissions_df.groupby("service")["admissions"].sum().reset_index()
        st.caption("Période vide : répartition sur l’ensemble des données.")
    else:
        part = adm_filtree.groupby("service")["admissions"].sum().reset_index()
    if services_choisis:
        part = part[part["service"].isin(services_choisis)]
    fig_pie = px.pie(part, values="admissions", names="service", title=f"Répartition du {date_debut_hist.strftime('%d/%m/%Y')} au {date_fin_hist.strftime('%d/%m/%Y')}")
    st.plotly_chart(fig_pie, use_container_width=True)

# --- Prévisions ---
elif section == "Prévisions":
    st.header("Prévisions des besoins")
    st.caption("Modèles : Holt-Winters (saisonnalité 7j), régression (lags 1–7–14 + mean j-7 à j-13, jours fériés, vacances, température synth. — réf. Bouteloup), durée de séjour saisonnière (réf. Lequertier) ; IC 95 %.")

    # Période de prévisions : date de début (fixe) + date de fin ou nombre de jours
    last_date = occupation_df["date"].max()
    start_prevision = last_date + pd.Timedelta(days=1)
    start_prevision_date = start_prevision.date() if hasattr(start_prevision, "date") else pd.Timestamp(start_prevision).date()
    default_end = start_prevision + pd.Timedelta(days=14)
    default_end_date = default_end.date() if hasattr(default_end, "date") else pd.Timestamp(default_end).date()
    max_end_date = (start_prevision + pd.Timedelta(days=90)).date() if hasattr((start_prevision + pd.Timedelta(days=90)), "date") else pd.Timestamp(start_prevision + pd.Timedelta(days=90)).date()

    st.subheader("Période des prévisions")
    col_period1, col_period2, col_period3 = st.columns(3)
    with col_period1:
        st.date_input(
            "Début des prévisions (1er jour après les données)",
            value=start_prevision_date,
            disabled=True,
            key="pred_debut",
        )
    with col_period2:
        date_fin_prevision = st.date_input(
            "Fin des prévisions (date)",
            value=default_end_date,
            min_value=start_prevision_date,
            max_value=max_end_date,
            key="pred_fin",
        )
    with col_period3:
        # Nombre de jours = (fin - début) + 1 pour inclure le dernier jour
        horizon = (pd.Timestamp(date_fin_prevision) - pd.Timestamp(start_prevision_date)).days + 1
        if horizon < 1:
            horizon = 14
            date_fin_prevision = default_end_date
        horizon = min(max(horizon, 1), 90)
        st.metric("Nombre de jours prévus", horizon)

    st.caption(f"Prévisions du **{start_prevision_date.strftime('%d/%m/%Y')}** au **{date_fin_prevision.strftime('%d/%m/%Y')}** ({horizon} jours)")

    besoins = predict_besoins(occupation_df, horizon_jours=horizon)
    pred_df = besoins["previsions"]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Taux d'occupation max prévu", f"{besoins['taux_max_prevu']:.0%}")
        if "taux_max_high" in besoins and besoins["taux_max_high"] != besoins["taux_max_prevu"]:
            st.caption(f"Borne haute (IC 95 %) : {besoins['taux_max_high']:.0%}")
        st.info(besoins["recommandation"])
    with col2:
        st.subheader("Prévision occupation (lits)")
        fig_pred = go.Figure()
        fig_pred.add_trace(
            go.Scatter(
                x=pred_df["date"],
                y=pred_df["occupation_lits_pred"],
                mode="lines+markers",
                name="Occupation prévue",
            )
        )
        if "occupation_lits_low" in pred_df.columns and "occupation_lits_high" in pred_df.columns:
            x = list(pred_df["date"]) + list(pred_df["date"][::-1])
            y = list(pred_df["occupation_lits_high"]) + list(pred_df["occupation_lits_low"][::-1])
            fig_pred.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    fill="toself",
                    fillcolor="rgba(0,100,200,0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="IC 95 %",
                )
            )
        fig_pred.add_hline(y=1800 * 0.85, line_dash="dash", line_color="orange")
        fig_pred.add_hline(y=1800 * 0.95, line_dash="dash", line_color="red")
        fig_pred.update_layout(height=350, title="Prévision occupation des lits")
        fig_pred.update_xaxes(tickformat="%d/%m/%Y")
        st.plotly_chart(fig_pred, use_container_width=True)

    cols_table = ["date", "occupation_lits_pred", "admissions_pred", "taux_occupation_pred", "alerte"]
    if "occupation_lits_low" in pred_df.columns:
        cols_table.insert(2, "occupation_lits_low")
        cols_table.insert(3, "occupation_lits_high")
    st.subheader("Détail des prévisions")
    pred_display = pred_df[[c for c in cols_table if c in pred_df.columns]].head(horizon).copy()
    pred_display["date"] = pd.to_datetime(pred_display["date"]).dt.strftime("%d/%m/%Y")
    st.dataframe(pred_display, use_container_width=True)

    with st.expander("Validation du modèle (réf. Bouteloup : critère ±10 %)"):
        adm_series = occupation_df.set_index("date")["admissions_jour"]
        eval_res = evaluate_forecast_pct_within_10(adm_series, validation_days=90)
        if eval_res.get("pct_within_10") is not None:
            st.metric("% de jours à ±10 %", f"{eval_res['pct_within_10']:.1%}")
            st.caption(f"Biais moyen : {eval_res.get('mean_error', 0):+.1f} (positif = surestimation). "
                      f"Sous-estimation {eval_res.get('pct_sous_estimation', 0):.1%} des jours, "
                      f"surestimation {eval_res.get('pct_surestimation', 0):.1%}.")
        else:
            st.caption(eval_res.get("message", "Non disponible"))

    st.subheader("Export")
    pred_export = pred_df[["date", "occupation_lits_pred", "admissions_pred", "taux_occupation_pred", "alerte"]].copy()
    pred_export["date"] = pd.to_datetime(pred_export["date"]).dt.strftime("%Y-%m-%d")
    csv_bytes = pred_export.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Télécharger les prévisions (CSV)",
        data=csv_bytes,
        file_name=f"previsions_{date_fin_prevision.strftime('%Y-%m-%d')}.csv",
        mime="text/csv",
        key="dl_previsions",
    )

# --- Modèle Boosting (apprentissage sur le passé) ---
elif section == "Modèle Boosting (apprentissage)":
    st.header("Modèle Boosting (XGBoost / Gradient Boosting)")
    st.caption(
        "Modèle qui **apprend du passé** : mêmes variables que la régression (lags 1–7–14, mean j-7 à j-13, "
        "calendrier, température synth.). XGBoost si installé, sinon GradientBoostingRegressor. "
        "Validation sur les derniers 90 jours (train sur le reste)."
    )
    adm_series = prepare_series(occupation_df, "admissions_jour")
    if adm_series.empty or len(adm_series) < 120:
        st.warning("Série d’admissions trop courte pour entraîner le modèle boosting (≥ 120 jours recommandés).")
    else:
        validation_jours = st.slider("Jours de validation (test)", 30, 120, 90, 15, key="boost_val_days")
        eval_boost = evaluate_boosting_model(adm_series, validation_days=validation_jours)
        if eval_boost.get("message"):
            st.info(eval_boost["message"])
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("MAE (admissions/jour)", f"{eval_boost['mae']:.1f}")
            with col2:
                st.metric("RMSE", f"{eval_boost['rmse']:.1f}")
            with col3:
                st.metric("% à ±10 % (Bouteloup)", f"{eval_boost['pct_within_10']:.1%}")
            with col4:
                st.metric("Biais moyen", f"{eval_boost['mean_error']:+.1f}")
            st.caption("Biais > 0 = surestimation en moyenne ; < 0 = sous-estimation.")
            vs = eval_boost.get("boosting_vs_best") or {}
            if vs:
                st.subheader("Comparaison Boosting vs modèle principal (Holt-Winters/Ridge)")
                st.json({
                    "MAE boosting": round(vs.get("mae_boosting"), 2),
                    "MAE modèle principal": round(vs.get("mae_best"), 2),
                    "RMSE boosting": round(vs.get("rmse_boosting"), 2),
                    "RMSE modèle principal": round(vs.get("rmse_best"), 2),
                    "Meilleur (MAE)": vs.get("meilleur_mae"),
                    "Meilleur (RMSE)": vs.get("meilleur_rmse"),
                })
        st.subheader("Prévision à 14 jours (modèle boosting)")
        pred_boost = predict_boosting(adm_series, horizon_jours=14)
        if pred_boost is not None and not pred_boost.empty:
            fig_boost = go.Figure()
            fig_boost.add_trace(
                go.Scatter(
                    x=pred_boost["date"],
                    y=pred_boost["prediction"],
                    mode="lines+markers",
                    name="Admissions (boosting)",
                )
            )
            if "prediction_low" in pred_boost.columns and "prediction_high" in pred_boost.columns:
                x = list(pred_boost["date"]) + list(pred_boost["date"][::-1])
                y = list(pred_boost["prediction_high"]) + list(pred_boost["prediction_low"][::-1])
                fig_boost.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        fill="toself",
                        fillcolor="rgba(0,150,80,0.2)",
                        line=dict(color="rgba(255,255,255,0)"),
                        name="IC 95 %",
                    )
                )
            fig_boost.update_layout(height=350, title="Prévision des admissions (modèle boosting)")
            fig_boost.update_xaxes(tickformat="%d/%m/%Y")
            st.plotly_chart(fig_boost, use_container_width=True)
            st.dataframe(
                pred_boost.assign(date=pd.to_datetime(pred_boost["date"]).dt.strftime("%d/%m/%Y")),
                use_container_width=True,
            )
        else:
            st.info("Prévision boosting non disponible (vérifier les dépendances : xgboost ou scikit-learn).")

# --- Simulation de scénarios ---
elif section == "Simulation de scénarios":
    st.header("Simulation de scénarios")
    st.info(
        "**Taux d'occupation effectif** (scénario grève) = occupation / **capacité effective**. "
        "La capacité effective baisse (lits fermés ou non utilisables). Un **taux ≥ 100 %** signifie que la demande dépasse les lits disponibles → **saturation**. "
        "La courbe n'est plus plate : montée en charge de l'effet grève (J1–J2), puis accumulation du flux (retards de sortie) sur la période."
    )
    scenario_choice = st.selectbox(
        "Choisir un scénario",
        list(SCENARIOS.keys()),
        format_func=lambda k: SCENARIOS[k].get("label", k),
    )
    duree_scenario_jours = st.slider(
        "Durée du scénario (jours)",
        min_value=14,
        max_value=90,
        value=30,
        step=7,
        help="Période simulée : plus longue pour voir l’évolution sur plusieurs semaines.",
    )

    if st.button("Lancer la simulation"):
        result = run_scenario(
            scenario_choice,
            occupation_df,
            admissions_df,
            duree_jours_override=duree_scenario_jours,
        )
        st.subheader(result["label"])
        st.metric("Taux d'occupation max estimé", f"{result['taux_occupation_max']:.0%}")
        st.metric("Niveau d'alerte", result["alerte"].upper())

        if not result["resultat"].empty:
            df = result["resultat"]
            if "taux_occupation_estime" in df.columns:
                fig = px.line(df, x="date", y="taux_occupation_estime", title="Taux d'occupation estimé (capacité normale)")
            elif "taux_occupation_effectif" in df.columns:
                fig = px.line(df, x="date", y="taux_occupation_effectif", title="Taux d'occupation effectif (capacité réduite)")
                fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="100 % = saturation")
            else:
                fig = px.line(df, x="date", y="admissions_scenario", title="Admissions simulées")
            fig.update_layout(height=400)
            fig.update_xaxes(tickformat="%d/%m/%Y")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df, use_container_width=True)
            df_export = df.copy()
            df_export["date"] = pd.to_datetime(df_export["date"]).dt.strftime("%Y-%m-%d")
            st.download_button(
                "Télécharger le scénario (CSV)",
                data=df_export.to_csv(index=False).encode("utf-8"),
                file_name=f"scenario_{scenario_choice}_{duree_scenario_jours}j.csv",
                mime="text/csv",
                key="dl_scenario",
            )

# --- Recommandations ---
elif section == "Recommandations":
    st.header("Recommandations automatiques")
    besoins = predict_besoins(occupation_df, horizon_jours=14)
    reco_forecast = recommendations_from_forecast(besoins["previsions"])

    st.subheader("À partir des prévisions")
    st.markdown(format_recommendations_for_dashboard(reco_forecast))

    st.subheader("À partir des scénarios")
    for sc_key, sc_conf in SCENARIOS.items():
        with st.expander(sc_conf.get("label", sc_key)):
            res = run_scenario(sc_key, occupation_df, admissions_df)
            reco_sc = recommendations_from_scenario(res)
            st.markdown(format_recommendations_for_dashboard(reco_sc))

st.sidebar.markdown("---")
with st.sidebar.expander("À propos des modèles (M2)"):
    st.caption(
        "**Prédiction** : Holt-Winters (saisonnalité 7j), régression Ridge (lags 1–7–14, mean j-7 à j-13, "
        "jours fériés, vacances, température synth. — réf. Bouteloup), durée de séjour saisonnière (réf. Lequertier). "
        "**Boosting** : XGBoost ou GradientBoosting (mêmes variables), apprentissage sur le passé, onglet dédié. "
        "Validation : critère ±10 %, biais (surestimation/sous-estimation). "
        "Justification détaillée : voir docs/01-rapport-conception/JUSTIFICATION-MODELES-PREDICTION.md. "
        "Pistes d’évolution : docs/PISTES-EVOLUTION.md"
    )
st.sidebar.caption("Projet Data Promo 2026 — Données fictives, pas de données réelles.")
