# Vue d’ensemble du projet — Pitié-Salpêtrière

Ce document donne une **explication globale** du projet, des **rapports** attendus et des **résultats** (chiffres, indicateurs, modèles) produits par le prototype. Il sert de point d’entrée pour comprendre l’ensemble et faire le lien entre livrables, dashboard et documentation technique.

---

## 1. Objectif et contexte

**Contexte** : L’Hôpital Pitié-Salpêtrière (Paris) doit anticiper les pics d’admission et optimiser ses ressources (lits, personnel, matériel). Le projet est un **MVP (Master 2)** de simulation et prévision des besoins hospitaliers, aligné sur les consignes officielles (projet Data Promo 2026).

**Objectifs** :
- Générer un **jeu de données fictif** inspiré de l’activité d’un grand hôpital (Pitié-Salpêtrière).
- Développer un **prototype fonctionnel** : tableau de bord interactif, prévisions (admissions, occupation), simulation de scénarios (épidémie, grève, canicule, afflux massif), recommandations automatiques.
- Produire un **rapport de conception** (méthodologie, modèles, analyse) et un **rapport stratégique** (freins/leviers, impact, comparaison, perspectives).
- Assurer la **conformité** (données de santé, RGPD, éthique) et s’appuyer sur la **littérature** (thèses Bouteloup 2020, Lequertier 2022).

**Données** : 100 % **synthétiques** (générées par `src.data.generate`). Aucune donnée réelle de patients ; le prototype est une démonstration dont la validation opérationnelle nécessiterait des données réelles (PMSI, RPU) dans un cadre réglementaire (CEREES, CNIL).

---

## 2. Architecture du projet et livrables

| Composant | Rôle | Où le trouver |
|-----------|------|----------------|
| **Données** | Jeu fictif (admissions par service, occupation quotidienne) | `data/generated/`, `src/data/` |
| **Prédiction** | Modèles (Holt-Winters, Ridge, SARIMA, MA, Boosting), modèle stock (occupation à partir des admissions) | `src/prediction/` |
| **Simulation** | Scénarios (épidémie, grève, canicule, afflux massif), durée paramétrable (14–90 j) | `src/simulation/` |
| **Recommandations** | Alertes et actions à partir des prévisions et des scénarios | `src/recommendations/` |
| **Dashboard** | Interface Streamlit : Flux & historique, Prévisions, Simulation, Boosting, Recommandations | `app/dashboard.py` |
| **Rapport de conception** | Fonctionnalités, méthodologie, tendances, analyse statistique, modèles, évaluation | [01-rapport-conception/](01-rapport-conception/) |
| **Rapport stratégique** | Freins/leviers, recommandations, impact, comparaison, perspectives | [02-rapport-strategique/](02-rapport-strategique/) |
| **Soutenance** | Plan 20 min, direction Pitié-Salpêtrière | [soutenance/](soutenance/) |

Référentiel des consignes (mapping PDF → livrables) : [05-reference/REFERENTIEL-CONSIGNES.md](05-reference/REFERENTIEL-CONSIGNES.md).

---

## 3. Modèles de prédiction et résultats affichés

### 3.1 Chaîne prévisionnelle

1. **Admissions** (volume journalier) : Holt-Winters, régression Ridge (lags 1–7–14, calendrier, splines), SARIMA, moyenne glissante, Boosting (XGBoost/GBM). Un seul modèle est utilisé à la fois (ordre : HW → Ridge → SARIMA → MA selon disponibilité ; Boosting dans un onglet dédié).
2. **Occupation des lits** : soit **prédiction directe** (Holt-Winters ou Ridge sur la série occupation_lits), soit **via admissions** (prédiction des admissions puis modèle stock avec durée de séjour saisonnière, réf. Lequertier).
3. **Besoins** : taux d’occupation prévu = occupation_lits_pred / 1800 ; seuils 85 % (alerte) et 95 % (critique) ; recommandations (renforts, vigilance, report d’activité).

Formules et cohérence mathématique : [03-modeles-et-resultats/COHERENCE-MODELES-PREVISION.md](03-modeles-et-resultats/COHERENCE-MODELES-PREVISION.md).  
Justification méthodologique (niveau M2) : [01-rapport-conception/JUSTIFICATION-MODELES-PREDICTION.md](01-rapport-conception/JUSTIFICATION-MODELES-PREDICTION.md).

### 3.2 Chiffres et indicateurs affichés (dashboard)

| Indicateur | Signification | Lien avec les modèles |
|------------|----------------|------------------------|
| **Taux d’occupation max prévu** | Max sur l’horizon de (occupation_lits_pred / 1800), en % | Sortie directe des modèles d’occupation (direct ou via admissions). |
| **Borne haute IC 95 %** | Max de la borne haute de l’intervalle de confiance | Tous les modèles fournissent une fourchette (low/high) ; utile pour planifier en prudent (réf. Bouteloup : privilégier la borne haute). |
| **% à ±10 %** (Bouteloup) | Proportion de jours où l’erreur relative (prévu vs observé) est ≤ 10 % | Métrique de **validation** : calculée sur une période de test (ex. 90 derniers jours) ; backtest et onglet Boosting. Référence littérature : Bouteloup ~84 % sur données réelles (Pellegrin). |
| **MAE, RMSE, biais moyen** | Erreur absolue moyenne, racine de l’erreur quadratique moyenne, moyenne (prévu − observé) | Métriques du backtest et de l’onglet Boosting ; biais > 0 = surestimation en moyenne. |
| **Sous-estimation / surestimation** | % de jours où prévu < 0,9×réel ou prévu > 1,1×réel | Complètent le % à ±10 % ; la sous-estimation est plus risquée pour la planification (Bouteloup). |
| **Seuils 85 % et 95 %** | Alerte et critique (constantes) | Appliqués au **taux d’occupation** prévu pour déclencher les recommandations ; pas une sortie de modèle. |

**Pourquoi les % sont souvent élevés** (taux d’occupation 65–85 %, % à ±10 % 85–95 %) : données synthétiques calibrées pour un hôpital chargé et séries lisses donc prévisibles ; sur données réelles, le % à ±10 % serait typiquement plus bas (70–85 %). Détail : [03-modeles-et-resultats/EXPLICATION-POURCENTAGES.md](03-modeles-et-resultats/EXPLICATION-POURCENTAGES.md).

### 3.3 Graphiques et exports

- **Flux & historique** : courbes admissions par service, taux d’occupation ; répartition (camembert) ; heatmaps (jour de la semaine × mois). Filtres : période, services, lissage 7 j, agrégation quotidienne/hebdomadaire.
- **Prévisions** : courbe occupation prévue (et IC 95 %), tableau détail (date, occupation, admissions si dispo, taux, alerte), backtest (prévu vs observé), export CSV.
- **Simulation** : courbe taux d’occupation (estimé ou effectif selon scénario), durée 14–90 j, export CSV.
- **Modèle Boosting** : métriques (MAE, RMSE, % ±10 %, biais), comparaison au modèle principal, prévision 14 j.

---

## 4. Rapports de résultats et où les trouver

| Thème | Contenu | Document / emplacement |
|-------|---------|-------------------------|
| **Résultats des modèles** | Formules, cohérence, quand les estimations se rapprochent ou divergent | [03-modeles-et-resultats/COHERENCE-MODELES-PREVISION.md](03-modeles-et-resultats/COHERENCE-MODELES-PREVISION.md) |
| **Interprétation des %** | Taux d’occupation, % à ±10 %, seuils, pourquoi “élevés” | [03-modeles-et-resultats/EXPLICATION-POURCENTAGES.md](03-modeles-et-resultats/EXPLICATION-POURCENTAGES.md) |
| **Justification des choix** | Familles de modèles, hypothèses, limites, protocole de validation | [01-rapport-conception/JUSTIFICATION-MODELES-PREDICTION.md](01-rapport-conception/JUSTIFICATION-MODELES-PREDICTION.md) |
| **Littérature et chiffres** | Bouteloup 83,79 % à ±10 % (Pellegrin), Batal −18,5 % départs sans soins, −30 % plaintes | [04-litterature/ANALYSE-CRITIQUE-LITTERATURE-THESES.md](04-litterature/ANALYSE-CRITIQUE-LITTERATURE-THESES.md), [04-litterature/VEILLE-THESES-DOCTORATS.md](04-litterature/VEILLE-THESES-DOCTORATS.md) |
| **Impact et adoption** | À traiter dans le rapport stratégique (freins/leviers, impact potentiel, comparaison) | [02-rapport-strategique/](02-rapport-strategique/) |
| **Pistes d’évolution** | Modèles (ensemble, météo, durée de séjour), scénarios, alertes, planification | [05-reference/PISTES-EVOLUTION.md](05-reference/PISTES-EVOLUTION.md) |

---

## 5. Synthèse : du prototype aux rapports

- **Dashboard** : reflète les **chiffres** (taux, % à ±10 %, MAE, RMSE, biais) et les **sorties des modèles** (prévisions, IC, alertes). Les données étant fictives, les niveaux (ex. taux 70 %, % ±10 % 88 %) illustrent le comportement des algorithmes et le calibrage du générateur, pas une performance sur du réel.
- **Rapport de conception** : décrit les **fonctionnalités**, la **méthodologie**, les **modèles** (avec justification dans [01-rapport-conception/JUSTIFICATION-MODELES-PREDICTION.md](01-rapport-conception/JUSTIFICATION-MODELES-PREDICTION.md)) et l’**évaluation** (métrique ±10 %, backtest). Les **résultats** à y rapporter sont ceux affichés dans le dashboard (ou exportés) et interprétés via [03-modeles-et-resultats/](03-modeles-et-resultats/).
- **Rapport stratégique** : s’appuie sur la **littérature** (Batal, Bouteloup, Lequertier) pour l’impact potentiel et la comparaison avec l’existant ; utilise les **chiffres du prototype** comme illustration, en rappelant que la validation opérationnelle nécessite des données réelles.

Pour naviguer dans toute la documentation par catégorie : [docs/README.md](README.md).

---

*Projet Data Pitié-Salpêtrière — Promo 2026. Données fictives.*
