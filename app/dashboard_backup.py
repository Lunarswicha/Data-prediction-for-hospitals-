"""
Tableau de bord interactif — MVP Pitié-Salpêtrière.
Exploration des flux, prévisions, simulation de scénarios, recommandations.
Lancer : streamlit run app/dashboard.py (depuis la racine du projet)
"""

import sys
import warnings
from pathlib import Path

# Éviter que les messages de dépréciation (Streamlit, pandas, plotly) s'affichent au-dessus des graphiques
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*use_container_width.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*keyword arguments have been deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Use config instead.*", category=UserWarning)

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
    predict_holt_winters,
    predict_regression,
    predict_sarima,
    predict_moving_average,
    predict_occupation_direct,
    predict_occupation_from_admissions,
    evaluate_forecast_pct_within_10,
    evaluate_boosting_model,
    predict_boosting,
    prepare_series,
    run_backtest_admissions,
    select_best_model_by_backtest,
)
from src.prediction.ensemble import predict_admissions_ensemble, get_ensemble_info


def _build_besoins_from_pred_df(pred_df: pd.DataFrame, capacite_lits: int = 1800) -> dict:
    """Construit le dict besoins (taux, alerte, reco) à partir d'un DataFrame de prévisions occupation."""
    if pred_df.empty:
        return {
            "previsions": pred_df,
            "taux_max_prevu": 0.0,
            "taux_max_high": 0.0,
            "recommandation": "Données insuffisantes.",
            "seuils": {"alerte": 0.85, "critique": 0.95},
        }
    p = pred_df.copy()
    if "taux_occupation_pred" not in p.columns:
        p["taux_occupation_pred"] = p["occupation_lits_pred"] / capacite_lits
        low_col = p.get("occupation_lits_low")
        high_col = p.get("occupation_lits_high")
        p["taux_occupation_low"] = (p["occupation_lits_pred"] if low_col is None else low_col) / capacite_lits
        p["taux_occupation_high"] = (p["occupation_lits_pred"] if high_col is None else high_col) / capacite_lits
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
        reco = "Vigilance : la borne haute des prévisions approche le seuil d'alerte."
    else:
        reco = "Capacité dans la norme ; maintenir la vigilance."
    return {
        "previsions": p,
        "taux_max_prevu": max_occ,
        "taux_max_high": max_high,
        "recommandation": reco,
        "seuils": {"alerte": seuil_alerte, "critique": seuil_critique},
    }


from src.simulation.scenarios import run_scenario, SCENARIOS
from src.recommendations.engine import (
    recommendations_from_forecast,
    recommendations_from_scenario,
    format_recommendations_for_dashboard,
)

st.set_page_config(
    page_title="MVP Pitié-Salpêtrière — Prévision des besoins",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Chemins possibles pour le logo (sidebar et header)
APP_DIR = Path(__file__).resolve().parent
ASSETS = APP_DIR / "assets"
LOGO_PNG = ASSETS / "logo_salpetriere.png"
LOGO_SVG = ASSETS / "logo_salpetriere.svg"
BATIMENT_PNG = ASSETS / "batiment_salpetriere.png"

# CSS pour améliorer l'esthétique
st.markdown("""
<style>
    /* En-tête principal */
    .main-header {
        padding: 0.75rem 0 1.25rem 0;
        border-bottom: 2px solid #0066a0;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        font-size: 1.75rem !important;
        font-weight: 600 !important;
        color: #0066a0 !important;
        margin-bottom: 0.25rem !important;
    }
    .main-header .caption {
        color: #64748b;
        font-size: 0.95rem;
    }
    /* Sidebar : espace autour du logo */
    [data-testid="stSidebar"] .logo-container {
        text-align: center;
        padding: 0.5rem 0 1rem 0;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid rgba(0,102,160,0.2);
    }
    [data-testid="stSidebar"] .logo-container img {
        max-width: 180px;
        height: auto;
    }
    [data-testid="stSidebar"] .logo-placeholder {
        font-size: 0.9rem;
        font-weight: 600;
        color: #0066a0;
        line-height: 1.3;
    }
    /* Sections : espacement */
    .stHeader { padding-top: 0.5rem; }
    div[data-testid="stVerticalBlock"] > div { padding: 0.1rem 0; }
    /* Métriques / indicateurs */
    [data-testid="stMetricValue"] { font-size: 1.25rem !important; }
    /* Bannière bâtiment */
    .banner-batiment {
        width: 100%;
        max-height: 200px;
        object-fit: cover;
        border-radius: 8px;
        margin-bottom: 1.25rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

try:
    admissions_df = load_admissions(ROOT / "data")
    occupation_df = load_occupation(ROOT / "data")
except FileNotFoundError as e:
    st.error(
        "Données non trouvées. Exécuter d'abord : `python -m src.data.generate` depuis la racine du projet."
    )
    st.stop()

# Bornes des données pour les filtres
_tmin = occupation_df["date"].min()
_tmax = occupation_df["date"].max()
date_min_data = pd.Timestamp(_tmin).date() if pd.notna(_tmin) else None
date_max_data = pd.Timestamp(_tmax).date() if pd.notna(_tmax) else None
if date_min_data is None or date_max_data is None:
    date_min_data = date_min_data or pd.Timestamp("2022-01-01").date()
    date_max_data = date_max_data or pd.Timestamp("2024-12-31").date()

# Sidebar — logo ou intitulé
if LOGO_PNG.exists():
    st.sidebar.image(str(LOGO_PNG), use_container_width=True)
elif LOGO_SVG.exists():
    with open(LOGO_SVG, encoding="utf-8") as f:
        st.sidebar.markdown(f.read(), unsafe_allow_html=True)
else:
    st.sidebar.markdown(
        "<div class='logo-placeholder'>Hôpital<br>Pitié-Salpêtrière</div>",
        unsafe_allow_html=True,
    )
st.sidebar.markdown("---")
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
    label_visibility="collapsed",
)

# En-tête principal (avec ou sans logo)
header_logo_html = ""
if LOGO_PNG.exists():
    import base64
    with open(LOGO_PNG, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    header_logo_html = f'<img src="data:image/png;base64,{b64}" alt="Logo" style="max-height:48px; margin-right:12px; vertical-align:middle;" /> '
st.markdown(
    f"""
    <div class="main-header">
        <span style="display:inline-flex; align-items:center; flex-wrap:wrap;">
            {header_logo_html}
            <h1 style="margin:0;">MVP Pitié-Salpêtrière</h1>
        </span>
        <p class="caption">Simulation et prévision des besoins hospitaliers — Données fictives</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Bannière : photo du bâtiment Pitié-Salpêtrière (si disponible)
if BATIMENT_PNG.exists():
    st.image(str(BATIMENT_PNG), use_container_width=True, output_format="PNG")

# --- Flux & historique ---
if section == "Flux & historique":
    st.header("Flux hospitaliers et historique")

    # Dernière année civile complète disponible dans les données
    year_max = date_max_data.year
    if date_max_data.month >= 12 and date_max_data.day >= 31:
        last_full_year = year_max
    else:
        last_full_year = year_max - 1
    year_min = date_min_data.year
    # Bornes des presets (clampées aux données)
    full_year_start = max(date_min_data, pd.Timestamp(last_full_year, 1, 1).date())
    full_year_end = min(date_max_data, pd.Timestamp(last_full_year, 12, 31).date())
    winter_start = max(date_min_data, pd.Timestamp(last_full_year - 1, 12, 1).date())
    winter_end = min(date_max_data, pd.Timestamp(last_full_year, 2, 28).date())
    summer_start = max(date_min_data, pd.Timestamp(last_full_year, 6, 1).date())
    summer_end = min(date_max_data, pd.Timestamp(last_full_year, 8, 31).date())

    st.subheader("Filtrer la période")
    preset = st.radio(
        "Type de période",
        options=["Année complète (dernière)", "Hiver (déc–fév)", "Été (juin–août)", "Personnalisée"],
        index=0,
        key="hist_preset",
        horizontal=True,
        help="Vue annuelle pour comparer pics hivernaux et baisse estivale ; hiver/été pour comparer les saisons.",
    )
    if preset == "Année complète (dernière)":
        default_debut, default_fin = full_year_start, full_year_end
    elif preset == "Hiver (déc–fév)":
        default_debut, default_fin = winter_start, winter_end
    elif preset == "Été (juin–août)":
        default_debut, default_fin = summer_start, summer_end
    else:
        _dmax = pd.Timestamp(date_max_data) if date_max_data is not None and pd.notna(date_max_data) else None
        default_debut = (_dmax - pd.DateOffset(months=12)).date() if _dmax is not None and pd.notna(_dmax) else date_min_data
        if default_debut < date_min_data:
            default_debut = date_min_data
        default_fin = date_max_data

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        date_debut_hist = st.date_input(
            "Du (date)",
            value=default_debut if (default_debut is not None and (not hasattr(pd, "isna") or not pd.isna(default_debut))) else date_min_data,
            min_value=date_min_data,
            max_value=date_max_data,
            key="hist_debut",
        )
    with col_f2:
        date_fin_hist = st.date_input(
            "Au (date)",
            value=default_fin if (default_fin is not None and (not hasattr(pd, "isna") or not pd.isna(default_fin))) else date_max_data,
            min_value=date_min_data,
            max_value=date_max_data,
            key="hist_fin",
        )
    with col_f3:
        st.caption("Vue annuelle : pics hiver, baisse été. Personnalisée : choisir les dates à la main.")

    if preset == "Année complète (dernière)" and full_year_start <= full_year_end:
        st.info(
            f"**Vue annuelle ({last_full_year})** — Les courbes ne suivent pas un pattern uniforme toute l'année : "
            "pics en hiver (grippe, bronchiolite) et baisse relative en été. Comparer avec « Hiver » et « Été » pour voir l'écart."
        )

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

    st.subheader("Répartition calendaire (heatmap)")
    st.caption("Moyenne des admissions (ou du taux d'occupation) par **jour de la semaine** et **mois**, sur la période filtrée.")
    jours_sem = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
    mois_noms = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
    col_hm1, col_hm2 = st.columns(2)
    with col_hm1:
        if not adm_filtree.empty:
            adm_daily = adm_filtree.groupby("date")["admissions"].sum().reset_index()
            adm_daily["date"] = pd.to_datetime(adm_daily["date"])
            adm_daily["mois"] = adm_daily["date"].dt.month
            adm_daily["jour_semaine"] = adm_daily["date"].dt.dayofweek
            adm_agg_cal = adm_daily.groupby(["mois", "jour_semaine"])["admissions"].mean().reset_index()
            pivot_adm = adm_agg_cal.pivot(index="mois", columns="jour_semaine", values="admissions")
            pivot_adm = pivot_adm.reindex(index=range(1, 13), columns=range(7)).fillna(0)
            z_adm = pivot_adm.values
            fig_hm_adm = go.Figure(
                data=go.Heatmap(
                    x=jours_sem,
                    y=mois_noms,
                    z=z_adm,
                    colorscale="Blues",
                    colorbar=dict(title="Admissions (moy.)"),
                )
            )
            fig_hm_adm.update_layout(
                title="Admissions (moyenne par jour de semaine × mois)",
                xaxis_title="Jour de la semaine",
                yaxis_title="Mois",
                height=400,
            )
            st.plotly_chart(fig_hm_adm, use_container_width=True)
        else:
            st.info("Aucune donnée pour la heatmap admissions.")
    with col_hm2:
        if not occ_filtree.empty:
            occ_cal = occ_filtree.copy()
            occ_cal["date"] = pd.to_datetime(occ_cal["date"])
            occ_cal["mois"] = occ_cal["date"].dt.month
            occ_cal["jour_semaine"] = occ_cal["date"].dt.dayofweek
            occ_agg_cal = occ_cal.groupby(["mois", "jour_semaine"])["taux_occupation"].mean().reset_index()
            pivot_occ = occ_agg_cal.pivot(index="mois", columns="jour_semaine", values="taux_occupation")
            pivot_occ = pivot_occ.reindex(index=range(1, 13), columns=range(7)).fillna(0)
            z_occ = pivot_occ.values * 100
            fig_hm_occ = go.Figure(
                data=go.Heatmap(
                    x=jours_sem,
                    y=mois_noms,
                    z=z_occ,
                    colorscale="Reds",
                    colorbar=dict(title="Taux (%)"),
                )
            )
            fig_hm_occ.update_layout(
                title="Taux d'occupation (moyenne par jour × mois)",
                xaxis_title="Jour de la semaine",
                yaxis_title="Mois",
                height=400,
            )
            st.plotly_chart(fig_hm_occ, use_container_width=True)
        else:
            st.info("Aucune donnée pour la heatmap occupation.")

# --- Prévisions ---
elif section == "Prévisions":
    st.header("Prévisions des besoins")
    st.caption("Modèles : Holt-Winters (saisonnalité 7j), régression Ridge avec **splines** (jour_semaine, jour_du_mois, température — approximation GAM, réf. Bouteloup), lags 1–7–14 + calendrier, durée de séjour saisonnière (Lequertier) ; IC 95 %. Veille thèses : docs/04-litterature/VEILLE-THESES-DOCTORATS.md")
    # ----- 1. Période -----
    last_date = occupation_df["date"].max()
    start_prevision = last_date + pd.Timedelta(days=1)
    start_prevision_date = start_prevision.date() if hasattr(start_prevision, "date") else pd.Timestamp(start_prevision).date()
    HORIZON_MAX_JOURS = 180
    st.subheader("1. Sur quelle période ?")
    preset_horizons = {"14 jours": 14, "1 mois": 30, "3 mois": 90, "6 mois": 180}
    preset_choisi = st.radio(
        "Portée",
        options=list(preset_horizons.keys()),
        index=0,
        key="pred_preset",
        horizontal=True,
        label_visibility="collapsed",
    )
    horizon = min(preset_horizons[preset_choisi], HORIZON_MAX_JOURS)
    end_prevision_date = (start_prevision + pd.Timedelta(days=horizon - 1)).date()
    if isinstance(end_prevision_date, pd.Timestamp):
        end_prevision_date = end_prevision_date.date()
    _start_str = start_prevision_date.strftime("%d/%m/%Y") if (hasattr(start_prevision_date, "strftime") and pd.notna(start_prevision_date)) else str(start_prevision_date)
    _end_str = end_prevision_date.strftime("%d/%m/%Y") if (hasattr(end_prevision_date, "strftime") and pd.notna(end_prevision_date)) else str(end_prevision_date)
    st.markdown(f"**Du {_start_str} au {_end_str}** — **{horizon} jours**")

    # ----- 2. Modèle -----
    st.subheader("2. Quel modèle ?")
    with st.expander("Aide — Données, modèles, % affichés, vagues"):
        st.markdown(
            "**Données** : le jeu est **synthétique** (généré par `src.data.generate`), avec des tendances réalistes mais lisses. "
            "Il n’y a pas de données réelles de patients.\n\n"
            "**Modèles** : chaque option force un algorithme précis. **Via admissions** = on prévoit d’abord les entrées quotidiennes, "
            "puis on en déduit l’occupation des lits (formule de stock). **Occupation directe** = on prévoit directement le nombre de lits occupés. "
            "Sur des données synthétiques lisses, **Holt-Winters** et **Ridge** peuvent donner des courbes proches ; "
            "la **moyenne glissante** est en général plus plate (moins de vagues).\n\n"
            "**Taux d’occupation** (ex. 67 %, 82 %) = lits occupés prévus / 1800. "
            "**% à ±10 %** (Bouteloup) = proportion de jours où l’erreur relative est ≤ 10 %. "
            "**85 % et 95 %** = seuils d’alerte et critique (constantes). Détail : **docs/03-modeles-et-resultats/EXPLICATION-POURCENTAGES.md**\n\n"
            "**D’où viennent les « vagues » ?** La prévision repose sur la **saisonnalité hebdomadaire** (jour de la semaine) : "
            "le même jour se répète toutes les 7 jours, donc les pics enchaînent (ex. 5, 12, 19, 26 du mois) — ce n’est pas un pic « mi-mois » au sens métier. "
            "Un effet **fin de mois** est pris en compte via les variables *jour du mois* et *fin de mois* dans la régression."
        )

    MODELES_PREVISION = [
        ("Auto (meilleur disponible)", "auto"),
        ("Holt-Winters (admissions puis occupation)", "holt_winters"),
        ("Régression Ridge (admissions puis occupation)", "ridge"),
        ("SARIMA (admissions puis occupation)", "sarima"),
        ("Moyenne glissante (admissions puis occupation)", "ma"),
        ("Boosting XGBoost/GBM (admissions puis occupation)", "boosting"),
        ("Occupation directe (Holt-Winters sur les lits)", "direct_hw"),
        ("Occupation directe (Régression sur les lits)", "direct_ridge"),
    ]
    modele_choisi_label = st.selectbox(
        "Modèle de prédiction",
        options=[m[0] for m in MODELES_PREVISION],
        index=0,
        key="choix_modele_prevision",
        help="Permet de visualiser les prévisions par modèle pour la démo et de comparer les sorties.",
    )
    modele_choisi = next(m[1] for m in MODELES_PREVISION if m[0] == modele_choisi_label)
    CAPACITE = 1800

    # Calcul des prévisions selon le modèle choisi (logique explicite pour que la courbe change bien)
    best_model_name = None
    best_model_metrics = None
    if modele_choisi == "auto":
        besoins = predict_besoins(occupation_df, horizon_jours=horizon, capacite_lits=CAPACITE)
        # Afficher quel modèle a été sélectionné par le benchmark (meilleur % ±10 %, puis MAE)
        adm_series_auto = prepare_series(occupation_df, "admissions_jour")
        val_days = min(90, max(28, len(adm_series_auto) // 3))
        best_model_name, best_model_metrics = select_best_model_by_backtest(adm_series_auto, validation_days=val_days)
    elif modele_choisi == "direct_hw":
        pred_df = predict_occupation_direct(occupation_df, horizon_jours=horizon, prefer="holt_winters")
        if not pred_df.empty:
            pred_df["taux_occupation_pred"] = pred_df["occupation_lits_pred"] / CAPACITE
            pred_df["taux_occupation_low"] = pred_df["occupation_lits_low"] / CAPACITE
            pred_df["taux_occupation_high"] = pred_df["occupation_lits_high"] / CAPACITE
        besoins = _build_besoins_from_pred_df(pred_df, CAPACITE)
    elif modele_choisi == "direct_ridge":
        pred_df = predict_occupation_direct(occupation_df, horizon_jours=horizon, prefer="regression")
        if not pred_df.empty:
            pred_df["taux_occupation_pred"] = pred_df["occupation_lits_pred"] / CAPACITE
            pred_df["taux_occupation_low"] = pred_df["occupation_lits_low"] / CAPACITE
            pred_df["taux_occupation_high"] = pred_df["occupation_lits_high"] / CAPACITE
        besoins = _build_besoins_from_pred_df(pred_df, CAPACITE)
    else:
        # Via admissions : on prédit les admissions avec le modèle choisi, puis on déduit l'occupation
        adm_series = prepare_series(occupation_df, "admissions_jour")
        pred_adm = None
        if modele_choisi == "holt_winters":
            pred_adm = predict_holt_winters(adm_series, horizon_jours=horizon)
        elif modele_choisi == "ridge":
            pred_adm = predict_regression(adm_series, horizon_jours=horizon)
        elif modele_choisi == "sarima":
            pred_adm = predict_sarima(adm_series, horizon_jours=horizon)
        elif modele_choisi == "ma":
            pred_adm = predict_moving_average(adm_series, horizon_jours=horizon)
        elif modele_choisi == "boosting":
            pred_adm = predict_boosting(adm_series, horizon_jours=horizon)
        if pred_adm is not None and not pred_adm.empty and len(pred_adm) >= horizon:
            pred_df = predict_occupation_from_admissions(
                occupation_df,
                horizon_jours=horizon,
                duree_sejour_moy=6.0,
                use_best_admissions=False,
                duree_sejour_saisonniere=True,
                pred_adm=pred_adm,
            )
            besoins = _build_besoins_from_pred_df(pred_df, CAPACITE)
        else:
            besoins = predict_besoins(occupation_df, horizon_jours=horizon, capacite_lits=CAPACITE)
            st.caption("Modèle demandé non disponible (série trop courte ou erreur) ; affichage Auto.")

    pred_df = besoins["previsions"]

    # ----- 3. Résultats -----
    st.subheader("3. Résultats")
    if modele_choisi == "auto" and best_model_name:
        st.caption(
            f"Modèle affiché : **{modele_choisi_label}** — modèle utilisé pour les admissions : **{best_model_name}** "
            f"(sélectionné par benchmark : meilleur % à ±10 % sur les {val_days} derniers jours)."
        )
        if best_model_metrics and best_model_metrics.get(best_model_name):
            with st.expander("Voir les métriques du benchmark (tous les modèles)"):
                for name, m in best_model_metrics.items():
                    st.text(f"{name}: % ±10 % = {m.get('pct_within_10', 0):.1%}, MAE = {m.get('mae', 0):.1f}, RMSE = {m.get('rmse', 0):.1f}")
    else:
        st.caption(f"Modèle affiché : **{modele_choisi_label}**.")

    # Graphique en premier (pleine largeur)
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
    fig_pred.update_layout(height=350, title=f"Occupation prévue — {modele_choisi_label}")
    fig_pred.update_xaxes(tickformat="%d/%m/%Y")
    st.plotly_chart(fig_pred, use_container_width=True)
    st.caption("Saisonnalité hebdo (jour de la semaine) + effet possible fin de mois. Les pics répétés = même jour de la semaine, pas une logique calendaire « mi-mois ».")

    # Indicateurs et recommandation
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        st.metric("Taux d'occupation max prévu", f"{besoins['taux_max_prevu']:.0%}")
    with col_k2:
        if "taux_max_high" in besoins and besoins["taux_max_high"] != besoins["taux_max_prevu"]:
            st.metric("Borne haute (IC 95 %)", f"{besoins['taux_max_high']:.0%}")
    st.info(besoins["recommandation"])

    cols_table = ["date", "occupation_lits_pred", "admissions_pred", "taux_occupation_pred", "alerte"]
    if "occupation_lits_low" in pred_df.columns:
        cols_table.insert(2, "occupation_lits_low")
        cols_table.insert(3, "occupation_lits_high")
    st.subheader("Détail des prévisions")
    pred_display = pred_df[[c for c in cols_table if c in pred_df.columns]].head(horizon).copy()
    pred_display["date"] = pd.to_datetime(pred_display["date"]).dt.strftime("%d/%m/%Y")
    if "admissions_pred" in pred_display.columns and pred_display["admissions_pred"].isna().all():
        pred_display["admissions_pred"] = "—"
        st.caption("**Admissions prédites (—)** : en mode *prédiction directe* (occupation des lits sans passer par les admissions), les admissions ne sont pas estimées par le modèle.")
    elif "admissions_pred" in pred_display.columns:
        pred_display["admissions_pred"] = pred_display["admissions_pred"].apply(
            lambda x: round(x, 1) if pd.notna(x) else "—"
        )
    st.dataframe(pred_display, use_container_width=True)

    with st.expander("Validation du modèle (réf. Bouteloup : critère ±10 %)"):
        adm_series = occupation_df.set_index("date")["admissions_jour"]
        if isinstance(adm_series, pd.DataFrame):
            adm_series = adm_series.squeeze()
        eval_res = evaluate_forecast_pct_within_10(pd.Series(adm_series) if not isinstance(adm_series, pd.Series) else adm_series, validation_days=90)
        if eval_res.get("pct_within_10") is not None:
            st.metric("% de jours à ±10 %", f"{eval_res['pct_within_10']:.1%}")
            st.caption(f"Biais moyen : {eval_res.get('mean_error', 0):+.1f} (positif = surestimation). "
                      f"Sous-estimation {eval_res.get('pct_sous_estimation', 0):.1%} des jours, "
                      f"surestimation {eval_res.get('pct_surestimation', 0):.1%}.")
        else:
            st.caption(eval_res.get("message", "Non disponible"))

    st.subheader("Backtest : prévision vs réel")
    st.caption(
        "Simulation sur le passé : le modèle est entraîné sur les données avant la période de test, "
        "puis on compare les prévisions aux valeurs observées (admissions quotidiennes). "
        "**Le backtest utilise une fenêtre fixe** (ex. 90 derniers jours) : il est **indépendant** de la période de prévision choisie en haut (14 jours, 6 mois, etc.)."
    )
    adm_series_bt = occupation_df.set_index("date")["admissions_jour"]
    if isinstance(adm_series_bt, pd.DataFrame):
        adm_series_bt = adm_series_bt.squeeze()
    adm_series_bt = pd.Series(adm_series_bt) if not isinstance(adm_series_bt, pd.Series) else adm_series_bt
    backtest_jours = st.slider(
        "Période de test (jours)",
        min_value=30,
        max_value=120,
        value=90,
        step=15,
        key="backtest_days",
        help="Nombre de jours en fin de série utilisés pour comparer prévision vs observé.",
    )
    backtest_res = run_backtest_admissions(adm_series_bt, validation_days=backtest_jours, use_best=True)
    if backtest_res.get("message"):
        st.warning(backtest_res["message"])
    else:
        bt_df = backtest_res["backtest_df"]
        metrics = backtest_res["metrics"]
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("MAE (admissions/jour)", f"{metrics['mae']:.1f}")
        with m2:
            st.metric("RMSE", f"{metrics['rmse']:.1f}")
        with m3:
            st.metric("% à ±10 % (Bouteloup)", f"{metrics['pct_within_10']:.1%}")
        with m4:
            st.metric("Biais moyen", f"{metrics['mean_error']:+.1f}")
        fig_bt = go.Figure()
        fig_bt.add_trace(
            go.Scatter(
                x=bt_df["date"],
                y=bt_df["observé"],
                mode="lines",
                name="Observé",
                line=dict(color="rgb(0,100,200)", width=2),
            )
        )
        fig_bt.add_trace(
            go.Scatter(
                x=bt_df["date"],
                y=bt_df["prévu"],
                mode="lines",
                name="Prévu (backtest)",
                line=dict(color="rgb(200,80,0)", width=2, dash="dash"),
            )
        )
        if "prévu_basse" in bt_df.columns and "prévu_haute" in bt_df.columns:
            x = list(bt_df["date"]) + list(bt_df["date"][::-1])
            y = list(bt_df["prévu_haute"]) + list(bt_df["prévu_basse"][::-1])
            fig_bt.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    fill="toself",
                    fillcolor="rgba(200,80,0,0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="IC 95 % prévu",
                )
            )
        fig_bt.update_layout(
            height=380,
            title="Backtest : admissions observées vs prévues",
            xaxis_title="Date",
            yaxis_title="Admissions / jour",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig_bt.update_xaxes(tickformat="%d/%m/%Y")
        st.plotly_chart(fig_bt, use_container_width=True)
        with st.expander("Voir les données du backtest"):
            bt_display = bt_df.copy()
            bt_display["date"] = pd.to_datetime(bt_display["date"]).dt.strftime("%d/%m/%Y")
            st.dataframe(bt_display, use_container_width=True)

    st.subheader("Export")
    export_cols = [c for c in ["date", "occupation_lits_pred", "admissions_pred", "taux_occupation_pred", "alerte"] if c in pred_df.columns]
    pred_export = pred_df[export_cols].copy()
    pred_export["date"] = pd.to_datetime(pred_export["date"]).dt.strftime("%Y-%m-%d")
    csv_bytes = pred_export.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Télécharger les prévisions (CSV)",
        data=csv_bytes,
        file_name=f"previsions_{end_prevision_date.strftime('%Y-%m-%d')}.csv",
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
                    "MAE boosting": round(vs.get("mae_boosting") or 0, 2),
                    "MAE modèle principal": round(vs.get("mae_best") or 0, 2),
                    "RMSE boosting": round(vs.get("rmse_boosting") or 0, 2),
                    "RMSE modèle principal": round(vs.get("rmse_best") or 0, 2),
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
        "**Boosting** : XGBoost ou GBM (mêmes variables + splines type GAM). **Régression** : splines sur jour/mois/temp. pour effets non linéaires (veille thèses). "
        "Validation : critère ±10 %, biais (surestimation/sous-estimation). "
        "Justification détaillée : voir docs/01-rapport-conception/JUSTIFICATION-MODELES-PREDICTION.md. "
        "Cohérence / complémentarité des modèles : docs/03-modeles-et-resultats/COHERENCE-MODELES-PREVISION.md. "
        "Explication des % (taux, % à ±10 %) : docs/03-modeles-et-resultats/EXPLICATION-POURCENTAGES.md. "
        "Pistes d’évolution : docs/05-reference/PISTES-EVOLUTION.md. Vue d'ensemble : docs/VUE-ENSEMBLE-PROJET.md"
    )
st.sidebar.caption("Projet Data Promo 2026 — Données fictives, pas de données réelles.")
