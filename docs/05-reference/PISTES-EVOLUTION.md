# Pistes d’évolution — MVP Pitié-Salpêtrière

Document de travail pour prolonger le projet au-delà du MVP (axes non encore couverts ou à renforcer).

---

## 1. Modèles et prédiction

- **Modèle Boosting (XGBoost / GBM)**  
  Déjà intégré : onglet « Modèle Boosting (apprentissage) » avec validation sur le passé (MAE, RMSE, % à ±10 %), comparaison au modèle principal, prévision 14 jours.  
  Pistes : réglage hyperparamètres (grid search), prédiction par service, agrégation modèle ensembliste (Ridge + Boosting).

- **Comparaison systématique des modèles**  
  Tableau récapitulatif : Holt-Winters, Ridge, SARIMA, Boosting sur les mêmes périodes de test (MAE, RMSE, MAPE, % ±10 %).  
  Graphique « prévisions vs réel » côte à côte pour plusieurs horizons (7, 14, 30 j).

- **Prévision de la durée de séjour (LOS)**  
  Référence Lequertier : modélisation individuelle (arbre, régression) pour anticiper la durée de séjour et affiner le modèle stock (occupation = f(admissions, LOS)).

- **Données météo réelles**  
  Remplacer la température synthétique par des données Météo-France (Paris) pour améliorer la régression et le boosting (effet canicule, vague de froid).

---

## 2. Simulation et scénarios

- **Durée longue des scénarios**  
  Déjà en place : slider 14–90 jours pour la simulation.  
  Pistes : courbes paramétrables (pic, asymétrie), scénarios combinés (grève + épidémie), incertitude (bandes de simulation).

- **Scénarios « what-if » interactifs**  
  Saisie manuelle : +X % d’admissions à partir de J, réduction de capacité en %, durée.  
  Visualisation immédiate du taux d’occupation et des seuils d’alerte.

---

## 3. Alertes et recommandations

- **Règles d’alerte configurables**  
  Seuils (taux d’occupation, dépassement de l’IC 95 %) et délai (J+1, J+7) paramétrables dans un fichier de config ou l’interface.  
  Historique des alertes déclenchées (dates, scénario, seuil franchi).

- **Recommandations par service**  
  Actuellement globales ; extension : recommandations par service (urgences, réa, etc.) selon occupation et prévisions par service.

- **Export des recommandations**  
  Export PDF/CSV des recommandations générées (date, type, priorité, actions).

---

## 4. Capacité et planification

- **Planification des lits et du personnel**  
  Objectif : à partir des prévisions et des seuils, proposer des effectifs cibles (ETP, renforts) et des créneaux de déprogrammation.  
  Contraintes : ratios lit/infirmier, plages de mobilisation.

- **Cartographie des lits**  
  Visualisation par service (carte ou barres) : capacité, occupation prévue, écart à la cible.  
  Indicateurs : taux d’occupation par service, tendance 7 j.

---

## 5. Données et conformité

- **Anonymisation et traçabilité**  
  Référentiel déjà décrit dans `CONFORMITE.md`.  
  Pistes : journal des accès, durée de rétention des exports, signature des exports (hash, horodatage).

- **Connecteurs données réelles**  
  En cas de passage sur données réelles (DP, SI hospitalier) : connecteurs sécurisés, agrégation uniquement, pas de données nominatives dans le dashboard.

---

## 6. Interface et exploitation

- **Export CSV**  
  Déjà en place : prévisions et résultat de scénario téléchargeables en CSV.  
  Pistes : export des séries historiques filtrées, format standardisé (colonnes, encodage).

- **Rafraîchissement et planification**  
  Rechargement périodique des données (Streamlit ou job externe), prévisions recalculées (ex. tous les matins) et mises en cache.

- **Tableau de bord « direction »**  
  Vue synthétique : indicateurs clés (taux max, alerte, tendance), graphique principal, dernières recommandations.  
  Une page = un écran pour réunion de pilotage.

---

## 7. Évaluation et recherche

- **Backtesting**  
  Rejouer les modèles sur des périodes passées (ex. 2022–2023) et comparer aux décisions réelles ou à un référentiel (occupation observée).

- **Publication et reproductibilité**  
  Notebooks Jupyter pour les expériences (features, modèles, métriques) ; versioning des jeux de données synthétiques ; rapport technique (méthodes, limites, références bibliographiques).

---

*Ce document peut être mis à jour au fil des livrables et des retours (équipe, encadrant, commanditaire).*
