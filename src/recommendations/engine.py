"""
Module de recommandations automatiques à partir des prévisions et scénarios.
Produit des alertes et suggestions d'ajustement (personnel, lits, matériel).
Niveau de détail adapté à un usage décisionnel (direction, pôle, régulation).

Littérature : Bouteloup (2020) — sous-estimation du flux plus risquée ; privilégier la borne haute de l'IC.
Batal et al. — adaptation du planning selon la prédiction : −18,5 % départs sans soins, −30 % plaintes.
"""

import pandas as pd
from typing import List, Dict, Any, Optional


def recommendations_from_forecast(
    previsions_df: pd.DataFrame,
    seuil_alerte: float = 0.85,
    seuil_critique: float = 0.95,
    capacite_lits: int = 1800,
) -> List[Dict[str, Any]]:
    """
    Génère des recommandations à partir d'un DataFrame de prévisions.
    Chaque recommandation inclut : niveau, titre, message détaillé, rationale, priorité, actions concrètes.
    """
    if previsions_df.empty:
        return []

    if "taux_occupation_pred" in previsions_df.columns:
        taux = previsions_df["taux_occupation_pred"]
    elif "occupation_lits_pred" in previsions_df.columns:
        taux = previsions_df["occupation_lits_pred"] / capacite_lits
    else:
        return []

    reco = []
    max_taux = taux.max()
    max_date = previsions_df.loc[taux.idxmax(), "date"] if hasattr(taux, "idxmax") else previsions_df["date"].iloc[-1]
    max_date_str = pd.Timestamp(max_date).strftime("%d/%m/%Y")

    # Borne haute de l'IC si disponible (réf. Bouteloup : privilégier pour la planification)
    taux_high = None
    if "taux_occupation_high" in previsions_df.columns:
        taux_high = previsions_df["taux_occupation_high"].max()

    if max_taux >= seuil_critique:
        reco.append({
            "niveau": "critique",
            "titre": "Occupation critique prévue",
            "message": (
                f"Le taux d'occupation prévu atteint **{max_taux:.0%}** vers le **{max_date_str}**, "
                f"au-dessus du seuil critique (**{seuil_critique:.0%}**). "
                "La capacité en lits et en personnel sera insuffisante pour absorber le flux attendu, "
                "avec un risque direct sur la qualité des soins et les délais de prise en charge."
            ),
            "rationale": (
                "La littérature (Batal et al., adaptation du planning à la prédiction) montre qu'anticiper "
                "les pics permet de réduire les départs sans soins et les plaintes. En situation critique, "
                "le report des interventions non urgentes et l'activation des lits de réserve limitent la saturation."
            ),
            "priorite": "Immédiate",
            "actions": [
                "Renforcer les effectifs soignants sur la période identifiée (planning, astreintes).",
                "Reporter les interventions programmées non urgentes dans la fenêtre à risque.",
                "Activer les lits de réserve et vérifier les capacités de réanimation.",
                "Alerter la régulation et les services d'aval (SSR, HAD) pour fluidifier les sorties.",
            ],
        })
    elif max_taux >= seuil_alerte and max_taux < seuil_critique:
        reco.append({
            "niveau": "alerte",
            "titre": "Période sous tension",
            "message": (
                f"Le taux d'occupation prévu atteint **{max_taux:.0%}** (seuil d'alerte **{seuil_alerte:.0%}**). "
                f"La borne haute de l'intervalle de confiance à 95 % est à **{taux_high:.0%}**." if taux_high is not None else ""
            ) + (
                " Une dégradation modérée des conditions de travail et des délais est possible sans mesure corrective."
            ),
            "rationale": (
                "Privilégier la borne haute des prévisions pour les décisions (Bouteloup) permet d'éviter "
                "la sous-estimation du flux, plus fréquente et plus risquée pour la planification."
            ),
            "priorite": "Haute",
            "actions": [
                "Surveiller les effectifs et préparer une montée en charge (renforts identifiés).",
                "Vérifier les stocks de matériel et consommables sur la période.",
                "Coordonner avec les urgences pour anticiper les entrées et les orientations.",
            ],
        })
    if max_taux < seuil_alerte:
        reco.append({
            "niveau": "normal",
            "titre": "Capacité dans la norme",
            "message": (
                f"Le taux maximal prévu sur la période est de **{max_taux:.0%}**, en dessous du seuil d'alerte (**{seuil_alerte:.0%}**). "
                "La capacité prévisionnelle reste compatible avec une prise en charge dans des conditions normales."
            ),
            "rationale": (
                "Maintenir une vigilance sur les indicateurs permet de détecter un dépassement précoce "
                "et d'ajuster les ressources si la borne haute de l'IC approche le seuil d'alerte."
            ),
            "priorite": "Vigilance",
            "actions": [
                "Maintenir le suivi des indicateurs (occupation, passages, délais).",
                "Conserver les procédures de montée en charge identifiées et testées.",
            ],
        })

    return reco


def recommendations_from_scenario(
    scenario_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Génère des recommandations à partir du résultat d'un scénario de simulation.
    Messages détaillés, rationale et actions concrètes par type de scénario.
    """
    alerte = scenario_result.get("alerte", "normal")
    label = scenario_result.get("label", scenario_result.get("scenario", ""))
    taux = scenario_result.get("taux_occupation_max", 0)

    if alerte == "critique":
        return [{
            "niveau": "critique",
            "titre": f"Scénario « {label} » — risque critique",
            "message": (
                f"En cas de réalisation du scénario **{label}**, le taux d'occupation (ou taux effectif) "
                f"pourrait atteindre **{taux:.0%}**, au-dessus du seuil critique (95 %). "
                "Cela correspond à une saturation des capacités disponibles (lits et/ou personnel), "
                "avec un impact direct sur la sécurité des soins et les délais."
            ),
            "rationale": (
                "Les scénarios de stress-test permettent d'anticiper les mesures à déployer en situation "
                "exceptionnelle (plan de crise, renforts, communication). Une préparation en amont réduit "
                "le temps de réaction et limite l'impact sur les patients."
            ),
            "priorite": "Anticipation crise",
            "actions": [
                "Définir ou mettre à jour le plan de crise associé à ce scénario.",
                "Identifier les renforts possibles (internes, autres sites, intérim) et les modalités d'activation.",
                "Prévoir une communication direction / ARS / régulation en cas de dépassement prolongé.",
            ],
        }]
    if alerte == "alerte":
        return [{
            "niveau": "alerte",
            "titre": f"Scénario « {label} » — vigilance",
            "message": (
                f"En cas de **{label}**, le taux d'occupation estimé pourrait atteindre **{taux:.0%}**, "
                "entre les seuils d'alerte (85 %) et critique (95 %). La marge de manœuvre est réduite ; "
                "une surveillance renforcée et des mesures préventives sont recommandées."
            ),
            "rationale": (
                "Anticiper ces situations permet de préparer les ajustements (effectifs, lits, flux) "
                "sans attendre le pic, en cohérence avec les résultats de la littérature sur l'adaptation "
                "du planning à la prédiction du flux."
            ),
            "priorite": "Haute",
            "actions": [
                "Renforcer la surveillance des indicateurs pendant la période à risque.",
                "Vérifier les stocks et les capacités de report (bloc, consultations).",
            ],
        }]
    return [{
        "niveau": "normal",
        "titre": f"Scénario « {label} »",
        "message": (
            f"Impact modéré du scénario **{label}** (taux max estimé **{taux:.0%}**). "
            "Les procédures habituelles restent adaptées ; maintenir la vigilance sur l'évolution des flux."
        ),
        "rationale": (
            "Les scénarios à impact modéré servent de référence pour comparer avec des situations "
            "plus dégradées et valider la robustesse des dispositifs de gestion des pics."
        ),
        "priorite": "Vigilance",
        "actions": [
            "Maintenir les procédures habituelles et le suivi des indicateurs.",
        ],
    }]


def format_recommendations_for_dashboard(reco_list: List[Dict[str, Any]]) -> str:
    """Format enrichi pour affichage dans le tableau de bord (titre, message, rationale, priorité, actions)."""
    if not reco_list:
        return "Aucune recommandation pour le moment."
    lines = []
    for r in reco_list:
        lines.append(f"### {r.get('titre', '')} — **{r.get('niveau', '')}**")
        lines.append("")
        lines.append(r.get("message", ""))
        lines.append("")
        if r.get("rationale"):
            lines.append("*Justification :* " + r.get("rationale", ""))
            lines.append("")
        if r.get("priorite"):
            lines.append(f"**Priorité :** {r.get('priorite', '')}")
            lines.append("")
        lines.append("**Actions recommandées :**")
        for a in r.get("actions", []):
            lines.append(f"- {a}")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines).strip()
