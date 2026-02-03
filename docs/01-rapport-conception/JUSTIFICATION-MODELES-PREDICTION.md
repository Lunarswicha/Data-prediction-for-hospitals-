# Justification des modèles de prédiction — Niveau M2

Ce document explicite les **choix méthodologiques** des modèles de prédiction du MVP (admissions, occupation, besoins), les **hypothèses** sous-jacentes, les **limites** et le **cadre de validation**, afin de répondre aux exigences d’un rendu de niveau Master 2.

---

## 1. Objectifs des modèles

Les modèles servent à :

1. **Prédire le flux d’admissions** (quotidien, global ou par service) sur un horizon de 1 à 90 jours, pour anticiper les pics d’activité (consignes projet ; littérature : Bouteloup 2020, Batal et al.).
2. **Prédire l’occupation des lits** (nombre de lits occupés, taux d’occupation), soit directement sur la série d’occupation, soit via un **modèle stock** à partir des admissions et d’une durée de séjour (réf. Lequertier 2022).
3. **Produire des indicateurs de besoin** (seuils d’alerte 85 % / 95 %, recommandations) pour la décision (planning, renforts, report d’activité).

Les sorties doivent être **interprétables** par la direction et **opérationnelles** (horizon court à moyen, intervalles de confiance lorsque possible).

---

## 2. Familles de modèles envisagées

| Famille | Avantages | Inconvénients | Positionnement dans le projet |
|--------|------------|---------------|-------------------------------|
| **Séries temporelles (Holt-Winters, SARIMA)** | Gestion explicite de la tendance et de la saisonnalité ; pas besoin de variables exogènes ; bien documenté pour le flux hospitalier (Bouteloup : GAM + lags ; littérature ARIMA). | Moins flexible pour intégrer calendrier (fériés, vacances) ; SARIMA coûteux en calcul et en identification. | Holt-Winters retenu pour saisonnalité 7j ; SARIMA en option (fallback). |
| **Régression (Ridge) avec variables calendrier et lags** | Intégration directe du calendrier (jour de semaine, mois, fériés, vacances) et des lags (Bouteloup : lags 7–13), interprétable. | Nécessite une série suffisamment longue ; risque de surajustement si trop de variables. | Modèle principal enrichi (lags 1–7–14, mean j-7 à j-13, fériés, vacances, température synthétique). |
| **Machine learning (réseaux de neurones, forêt aléatoire)** | Puissant pour relations non linéaires ; Lequertier utilise un réseau de neurones pour la durée de séjour. | Moins interprétable ; besoin de beaucoup de données et de variables ; notre cible est agrégée (flux journalier), pas individuelle. | Non retenu pour le flux agrégé dans ce MVP ; à envisager si passage à des données patient (type PMSI). |

**Choix global** : privilégier des modèles **parsimonieux et interprétables** (Holt-Winters, régression Ridge), avec **intervalles de confiance** (IC 95 %) et **validation sur critère opérationnel** (±10 %, Bouteloup), tout en restant cohérent avec la littérature (Bouteloup, Lequertier).

---

## 3. Justification de chaque modèle

### 3.1 Lissage exponentiel de Holt-Winters (saisonnalité additive)

- **Rôle** : prédiction des admissions (et éventuellement de l’occupation) en capturant une **tendance** et une **saisonnalité hebdomadaire** (période 7 jours).
- **Justification** : le flux aux urgences et l’occupation présentent une forte **saisonnalité hebdo** (lundi et week-end plus chargés ; Bouteloup, littérature). Le modèle est simple, robuste et fournit des prévisions lissées sans variables exogènes.
- **Hypothèses** : saisonnalité additive ; tendance additive ; erreurs non corrélées. La série est supposée suffisamment longue (≥ 2 périodes saisonnières).
- **Limites** : ne modélise pas explicitement les jours fériés ni les vacances ; une seule saisonnalité (7j), pas de saisonnalité annuelle dans ce modèle.

### 3.2 Régression Ridge (lags + calendrier + température synthétique)

- **Rôle** : prédiction des admissions (et de l’occupation en mode direct) à partir de **variables explicatives** : lags 1, 7, 14, **moyenne j-7 à j-13** (Bouteloup), jour de semaine, mois, **jours fériés, veille/lendemain de férié, vacances scolaires zone C** (Bouteloup), **température synthétique** (proxy météo, Bouteloup).
- **Justification** : la littérature (Bouteloup) montre qu’un **GAM avec calendrier et lags 7–13** atteint environ **84 % de jours à ±10 %** sur le flux urgences. Une régression linéaire Ridge avec les mêmes types de variables permet de rester interprétable tout en régularisant (éviter le surajustement). La température est un proxy pour canicule/vague de froid.
- **Hypothèses** : relation approximativement linéaire entre les variables et la cible ; régularisation L2 (Ridge) ; variables centrées-réduites pour la stabilité.
- **Limites** : linéarité ; la température est synthétique (courbe sinusoïdale), pas de données Météo France dans ce MVP.

### 3.3 SARIMA (optionnel)

- **Rôle** : alternative pour séries longues avec saisonnalité 7 jours ; ordre (1,0,1)(1,0,1,7).
- **Justification** : utilisé dans la littérature pour le flux aux urgences (séries temporelles). Permet de capturer autocorrélation et saisonnalité.
- **Hypothèses** : stationnarité après différenciation ; ordre et ordre saisonnier fixés pour limiter le temps de calcul.
- **Limites** : coût d’estimation ; risque de non-convergence ; utilisé en fallback seulement.

### 3.4 Moyenne glissante + tendance (baseline)

- **Rôle** : prédiction de dernier recours lorsque la série est trop courte ou que les autres modèles échouent.
- **Justification** : fournit une prévision simple et stable ; sert de **référence** pour comparer les modèles plus élaborés.
- **Limites** : pas de saisonnalité ; pas de calendrier ; intervalles de confiance approximatifs.

### 3.5 Modèle stock (occupation à partir des admissions)

- **Rôle** : déduire l’**occupation des lits** à partir des **admissions prédites** et d’une **durée de séjour** (moyenne ou saisonnière).
- **Justification** : Lequertier (2022) montre que la **durée de séjour** est un levier central pour l’occupation ; une durée **variable** (saisonnière : hiver +8 %, été −8 %) améliore le réalisme par rapport à une durée fixe.
- **Hypothèses** : relation stock ≈ admissions × durée de séjour (modèle simplifié) ; durée moyenne ou saisonnière connue/estimée.
- **Limites** : modèle agrégé ; pas de prédiction individuelle de la durée de séjour (comme dans Lequertier avec NN/PMSI).

---

## 4. Protocole de validation et métriques

- **Critère opérationnel** : **% de jours à ±10 %** (écart relatif entre prédit et réel), en cohérence avec Bouteloup (83,79 % sur Pellegrin avec modèle lag).
- **Métriques complémentaires** : biais moyen (surestimation / sous-estimation), % de jours en surestimation vs sous-estimation. La **sous-estimation** est considérée comme plus risquée pour la planification (Bouteloup) ; les recommandations privilégient la **borne haute** de l’IC 95 % lorsque disponible.
- **Validation sur données fictives** : le MVP utilise un jeu de données **généré** ; la métrique ±10 % et le biais sont calculés sur une période de validation (ex. 90 derniers jours) pour illustrer le protocole. Une validation opérationnelle nécessiterait des **données réelles** (PMSI, RPU) dans un cadre réglementaire (CEREES, CNIL).

---

## 5. Hypothèses et limites globales

- **Données** : fictives ; pas de données réelles patients (PMSI, RPU). Les conclusions sur les performances ne sont pas généralisables sans validation sur données réelles.
- **Périmètre** : flux agrégé (admissions, occupation) ; pas de prédiction individuelle de la durée de séjour ni des hospitalisations non programmées (Bouteloup souligne l’intérêt de cette cible pour les lits d’aval).
- **Modèles** : choix volontairement parsimonieux et interprétables ; pas de comparaison systématique avec d’autres modèles (ARIMA, GAM, NN) dans ce livrable, mais cadre posé pour une telle comparaison (métrique ±10 %, biais).

---

## 6. Références

- **Bouteloup F.** (2020). Création et validation d’un modèle de prédiction du nombre de passages journaliers dans le service des urgences de l’Hôpital Pellegrin. Thèse médecine, Bordeaux. HAL dumas-03085214.
- **Lequertier V.** (2022). Méthode globale de prédiction des durées de séjours hospitalières avec intégration des données incrémentales et évolutives. Thèse Lyon 1. HAL tel-04053390.
- **Batal et al.** Predicting patient visits to an urgent care clinic using calendar variables. Acad Emerg Med. 2001.

---

*Document à intégrer au rapport de conception (livrable 1). Projet Data Pitié-Salpêtrière — Promo 2026.*
