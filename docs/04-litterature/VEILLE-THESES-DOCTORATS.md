# Veille — Thèses et doctorats (prévision flux hospitaliers, urgences, occupation)

Ce document complète l’analyse des thèses déjà intégrées (Bouteloup 2020, Lequertier 2022) par une veille ciblée et en tire des **améliorations du modèle** mises en œuvre dans le projet.

---

## 1. Références identifiées

### 1.1 Travaux déjà intégrés au projet

| Auteur | Année | Titre / contexte | Apport principal |
|--------|--------|------------------|------------------|
| **Florent Bouteloup** | 2020 | Création et validation d’un modèle de prédiction du nombre de passages journaliers aux urgences, Hôpital Pellegrin (Bordeaux) | GAM avec **calendrier + lags j-7 à j-13 + météo** ; critère **±10 %** ; ~84 % des jours corrects ; [Dumas 03085214](https://dumas.ccsd.cnrs.fr/dumas-03085214v1/document). |
| **Vincent Lequertier** | 2022 | Prédiction des durées de séjour hospitalières (données PMSI, HCL) | Réseau de neurones / forêt aléatoire ; **durée de séjour** prédictible ; impact sur occupation et besoins en lits ; [HAL tel-04053390](https://theses.hal.science/tel-04053390). |

### 1.2 Thèses et travaux complémentaires (veille)

| Auteur / étude | Année | Contexte | Résultat / méthode |
|----------------|--------|----------|---------------------|
| **Mathias Wargon** | 2010 | Modélisation et prédiction des flux aux services d’urgence (région parisienne, 2004-2007) | **Modèle linéaire + variables calendaires** : EMAP < 10 % ; forte influence du **jour de la semaine** ; pas de saisonnalité annuelle ; agrégation multi-sites améliore (EMAP 5,3 % vs 8–17 % par centre). [theses.fr 2010PA066547](https://theses.fr/2010PA066547) |
| **Sonia Rafi** | 2023 | Prédiction par apprentissage automatique des événements critiques aux urgences | Forêts aléatoires pour prédire décès, intubation, RCP, soins palliatifs ; [HAL tel-04506674](https://theses.hal.science/tel-04506674v1). |
| **Félicien Hêche** | 2024 | Risk-sensitive machine learning for emergency medical resource optimization | ML sensible au risque pour l’optimisation des ressources médicales d’urgence ; [HAL tel-04916963](https://theses.hal.science/tel-04916963v1). |
| **Étude multicentrique française** | 2024 | Deux services d’urgence (2010-2019), prédiction des admissions | **XGBoost** avec optimisation hyperparamètres : MAE 2,63–2,64 patients/heure ; [BMC Emergency Medicine](https://link.springer.com/article/10.1186/s12873-024-01141-4). |
| **Prédiction probabiliste Île-de-France** | 2025 | Prédiction des arrivées et hospitalisations aux urgences, plusieurs hôpitaux | Méthodes **ensemblistes** ; prédictions ponctuelles et **intervalles** pour la variabilité de la demande ; [HAL hal-04539380](https://hal.science/hal-04539380v1). |
| **NLP et flux urgences Bordeaux** | 2025 | Étude des flux et des risques traumatiques à partir des lieux de prise en charge | NLP pour analyser les flux et les motifs ; [HAL tel-04554258](https://theses.hal.science/tel-04554258v1). |

---

## 2. Synthèse pour la modélisation

- **Variables calendaires et lags** : Wargon et Bouteloup confirment l’intérêt du **jour de la semaine** et des **lags** (7–13) ; notre modèle est aligné (lags 1–7–14, mean j-7 à j-13, calendrier).
- **Effets non linéaires** : Bouteloup utilise des **GAM** ; la littérature souligne des relations **non linéaires** (jour de la semaine, météo). **Amélioration** : approximation par **splines** (SplineTransformer) sur jour_semaine, jour_du_mois, température pour approcher un effet type GAM.
- **Métriques** : EMAP / % à ±10 % (Wargon, Bouteloup) ; nous utilisons déjà le critère ±10 % et le backtest.
- **Ensembles et XGBoost** : travaux récents (Île-de-France, multicentrique) privilégient **ensembles** et **XGBoost** ; nous avons XGBoost/GBM et pouvons proposer un **ensemble** (moyenne Ridge + Boosting).
- **Asymétrie des erreurs** : Bouteloup signale une **sous-estimation** plus problématique pour la planification ; prise en compte via IC 95 % et recommandations sur la borne haute.
- **Fin de mois / calendrier** : Wargon insiste sur les **facteurs calendaires** ; nous avons ajouté **jour_du_mois** et **fin_mois** pour permettre un effet fin de mois.

---

## 3. Améliorations du modèle (implémentées ou proposées)

| Amélioration | Source | Statut |
|--------------|--------|--------|
| Lags 7–13 (moyenne) + calendrier (fériés, vacances) | Bouteloup | Déjà en place |
| Critère ±10 % et backtest prévision vs réel | Bouteloup, Wargon | En place |
| Variables jour_du_mois, fin_mois | Veille (effet calendaire) | En place |
| **Splines sur jour_semaine, jour_du_mois, température** (approximation GAM) | Bouteloup, littérature GAM | Ajouté (option `use_splines`) |
| XGBoost / Gradient Boosting, onglet dédié | Études récentes, multicentrique | En place |
| Heatmap répartition calendaire (jour × mois) | Demande projet | Ajoutée (dashboard) |
| Météo réelle (Météo France) | Bouteloup | Piste (stub possible) |
| Ensemble Ridge + Boosting pour admissions | Île-de-France 2025 | Piste (05-reference/PISTES-EVOLUTION.md) |

---

## 4. Références bibliographiques (forme courte)

- Bouteloup F. (2020). *Création et validation d'un modèle de prédiction du nombre de passages journaliers dans le service des urgences de l'Hôpital Pellegrin*. Thèse d’exercice, CHU Bordeaux. Dumas 03085214.
- Wargon M. (2010). *Modélisation et prédiction des flux aux services d’urgence*. Thèse, Paris. theses.fr 2010PA066547.
- Lequertier V. (2022). *Méthode globale de prédiction des durées de séjours hospitalières…* HAL tel-04053390.
- Predicting emergency department admissions using a machine-learning algorithm (2024). *BMC Emergency Medicine*.
- Probabilistic Prediction of Arrivals and Hospitalizations in Emergency Departments in Île-de-France (2025). *International Journal of Medical Informatics*, HAL hal-04539380.

---

*Document mis à jour avec la veille et les améliorations du modèle (splines, heatmap).*
