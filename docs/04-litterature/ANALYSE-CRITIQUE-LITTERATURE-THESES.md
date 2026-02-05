# Analyse critique — Littérature (thèses intégrées)

Ce document analyse précisément les deux thèses intégrées au projet et en tire un regard critique sur notre MVP, ainsi que des constats à intégrer dans les rapports et le prototype.

---

## 1. Synthèse des deux thèses

### 1.1 Thèse 1 — Vincent Lequertier (2022)

**Titre** : *Méthode globale de prédiction des durées de séjours hospitalières avec intégration des données incrémentales et évolutives*  
**Source** : HAL tel-04053390, Université Claude Bernard Lyon 1, RESHAPE / DISP.  
**Discipline** : Épidémiologie, santé publique, recherche sur les services de santé.

#### Objectifs
- Prédire la **durée de séjour** (nombre de jours entre admission et sortie) pour **tous les patients**, à **toutes les étapes** du parcours (unités médicales), en utilisant des données **médico-administratives standardisées** (PMSI MCO, RPU).
- Améliorer la gestion des lits, la préparation des sorties et l’organisation des services.

#### Méthodes
- **Revue systématique** de la littérature sur les méthodes de prédiction des durées de séjour (préparation des données, approches, reporting).
- **Prétraitement** des données et **embeddings** pour représenter les concepts médicaux.
- **Réseau de neurones** (feed-forward avec embeddings) pour prédire la durée de séjour ; comparaison avec **forêt aléatoire** et **régression logistique**.
- Données : **Hospices Civils de Lyon (HCL)** — RSS, RUM, GHM, RPU ; CIM-10, CCAM ; autorisations CEREES et CNIL.

#### Constats clés
- La durée de séjour peut être **prédite par un réseau de neurones** avec des données PMSI disponibles pour tous les patients.
- Les **prédictions du personnel médical** ne sont pas toujours exactes ; la durée de séjour dépend de nombreux facteurs (historiques, médicaux, environnementaux) difficiles à intégrer manuellement.
- Une **estimation uniquement basée sur les moyennes par diagnostic** (GHM) n’est pas suffisamment précise ; prendre en compte les **variations liées aux caractéristiques des patients et à la prise en charge** améliore la planification et la gestion des lits.
- La durée de séjour est importante pour le **taux d’occupation** et l’**estimation des besoins en lits et en personnels** ; les prédictions peuvent alimenter des **modèles de simulation** et des programmes de planification.
- Données **incrémentales et évolutives** : la prédiction doit s’adapter au fur et à mesure du parcours (données censurées pour la sortie au moment de l’admission dans une UM).

#### Limites (rapportées par l’auteur)
- Nécessité d’autorisations et d’un cadre réglementaire strict (CEREES, CNIL).
- Données PMSI/RPU spécifiques au contexte français et à la structure des établissements.

---

### 1.2 Thèse 2 — Florent Bouteloup (2020)

**Titre** : *Création et validation d’un modèle de prédiction du nombre de passages journaliers dans le service des urgences de l’Hôpital Pellegrin*  
**Source** : HAL dumas-03085214, Thèse d’exercice en médecine, CHU Bordeaux.  
**Contexte** : Service des urgences adultes, Hôpital Pellegrin (Bordeaux Métropole).

#### Objectifs
- Prédire le **nombre de passages journaliers aux urgences** pour aider à organiser les ressources (personnel, lits d’hospitalisation non programmés).
- Créer et valider un modèle utilisant des **données calendaires** et **météorologiques**, ainsi que le **flux des jours précédents**.

#### Méthodes
- **Modèle additif généralisé (GAM)** avec trois variantes :
  1. **Calendrier** : jour de la semaine, mois, jours fériés, veilles/lendemains, vacances scolaires (zone A).
  2. **Calendrier + lag** : ajout du nombre de passages des **jours 7 à 13** précédant chaque jour.
  3. **Calendrier + lag + météo** : ajout de 11 variables météo (températures min/max, pression, vent, humidité, précipitations, nébulosité).
- Données : **1er janvier 2010 – 31 décembre 2018** pour l’apprentissage ; **2019** pour la validation. Source : Business Objects®, Météo France®.
- **Critère de succès** : prédiction considérée correcte si l’écart entre prédit et réel est **≤ 10 %**.

#### Résultats
- **Modèle lag** (calendrier + passages j-7 à j-13) : **83,79 %** des jours corrects à 10 % près sur 2019.
- **Sous-estimation** plus fréquente que la surestimation : **11,54 %** sous-estimation vs **4,67 %** surestimation (modèle lag). Pour la planification, une sous-estimation du flux est plus problématique (risque de sous-dimensionnement).
- **Lundi et week-end** : plus de passages ; **jours fériés** : flux plus élevé (159 vs 147 en médiane).
- Les **variables météorologiques** améliorent légèrement le R² (0,505 → 0,528) et la corrélation de Pearson, mais le modèle « météo » obtient un taux de jours corrects légèrement inférieur (79,33 %) — données météo manquantes pour certains jours, impact variable selon la région.
- **Bland & Altman** : dispersion plus homogène pour le modèle météo ; tendance à sous-estimer pour les flux faibles et à surestimer pour les flux élevés.

#### Constats clés
- Un **GAM avec calendrier + lags (7–13)** permet une prédiction opérationnelle du flux journalier aux urgences (≈ 84 % des jours à ±10 %).
- La **littérature** cite d’autres variables (vacances scolaires, épidémies type grippe) et d’autres modèles (séries temporelles, ARIMA, régression linéaire généralisée, réseaux de neurones) ; le choix des variables doit tenir compte du **contexte géographique et du type de centre**.
- **Batal et al.** : adapter le planning du personnel selon la prédiction du flux a permis une **diminution de 18,5 %** des patients partis sans soins et de **30 %** des plaintes.
- **Limites** : étude monocentrique ; pas de comparaison avec ARIMA ou autres modèles ; pas d’accès au **nombre d’hospitalisations non programmées** (indicateur plus pertinent pour les lits d’aval) ; pas de catégorisation des patients (âge, gravité). L’auteur souligne qu’il serait plus pertinent de prédire les **hospitalisations non programmées** pour fluidifier l’activité et la mise à disposition des lits.

---

## 2. Positionnement de notre projet par rapport aux thèses

| Dimension | Thèse Lequertier | Thèse Bouteloup | Notre MVP (Pitié-Salpêtrière) |
|-----------|------------------|------------------|-------------------------------|
| **Cible de prédiction** | Durée de séjour **individuelle** (par patient, par RUM) | **Volume journalier** (passages urgences) | **Volume journalier** (admissions par service) + **occupation des lits** (agrégée) |
| **Niveau** | Patient / unité médicale | Établissement (un service) | Établissement (multi-services, fictif) |
| **Données** | PMSI (RSS, RUM), RPU, CIM-10, CCAM — réelles HCL | Passages urgences réels + calendrier + météo | **Données fictives** (générées), tendances réalistes |
| **Modèles** | Réseau de neurones, forêt aléatoire, régression logistique | GAM (calendrier, lag 7–13, météo) | Holt-Winters, régression (lags 1–7–14 + calendrier), SARIMA, moyenne glissante |
| **Validation** | Comparaison de modèles sur données HCL | 2019 hors échantillon, critère ±10 % | Pas de données réelles ; scénarios de simulation |
| **Usage** | Aide à la planification des lits et des sorties | Planification personnel et lits non programmés | Tableau de bord, prévisions, scénarios (épidémie, grève, etc.), recommandations |

**Convergence avec Bouteloup** : notre approche est du même type que la sienne — **prédiction du flux journalier** (admissions / occupation) avec **variables calendaires et lags** ; les constats sur le jour de la semaine (lundi, week-end), l’intérêt des lags et la prudence sur la sous-estimation s’appliquent directement.

**Complémentarité avec Lequertier** : il traite la **durée de séjour individuelle** et les données **PMSI/RPU** ; nous ne modélisons pas la durée de séjour au niveau patient, mais nous utilisons une **durée de séjour moyenne** dans le modèle stock (occupation à partir des admissions). Son travail justifie qu’une **meilleure prédiction de la durée de séjour** (éventuellement par type de séjour ou unité) pourrait renforcer notre chaîne prévisionnelle.

---

## 3. Regard critique sur notre travail — Constats des thèses

### 3.1 Points forts (alignés avec la littérature)
- **Variables calendaires et lags** : nous utilisons jour de la semaine, mois et lags (1, 7, 14), en cohérence avec Bouteloup (lags 7–13, calendrier) et avec la littérature sur le flux aux urgences.
- **Saisonnalité** : Holt-Winters avec période 7 jours capture une saisonnalité hebdomadaire, pertinente pour les urgences et l’occupation.
- **Intervalles de confiance** : nous fournissons des fourchettes (IC 95 %) ; Bouteloup utilise un seuil ±10 % pour qualifier une prédiction « correcte » — nous pouvons discuter un critère opérationnel similaire (ex. % de jours dans l’IC).
- **Scénarios** : épidémie, grève, canicule, afflux massif — en résonance avec la littérature (épidémies, événements exceptionnels) et avec les besoins de la direction.
- **Tableau de bord et recommandations** : objectif proche de Batal et al. / Bouteloup — aider à ajuster personnel et ressources.

### 3.2 Limites et pistes d’amélioration (tirées des thèses)

#### Données
- **Données fictives** : les deux thèses s’appuient sur des **données réelles** (HCL, Pellegrin). Notre MVP ne peut pas prétendre à une validation clinique ou opérationnelle ; il doit être présenté comme un **prototype de démonstration** dont la validation nécessiterait l’accès à des données réelles (PMSI, RPU, etc.) dans un cadre réglementaire (CEREES, CNIL, référentiel HDS).
- **PMSI / structure fine** : Lequertier montre l’intérêt des **RUM, GHM, CIM-10, CCAM** pour une prédiction fine. Notre jeu fictif est agrégé (admissions par service, occupation globale) ; une évolution vers des données de type PMSI (anonymisées) permettrait d’enrichir les modèles et, si besoin, de tendre vers une prédiction de la durée de séjour par segment.

#### Modèles
- **Durée de séjour** : nous utilisons une **durée de séjour moyenne fixe** (ex. 6 jours) dans le modèle stock. Lequertier montre que la **durée de séjour est prédictible** (réseau de neurones, forêt aléatoire) à partir de données médico-administratives ; une **durée de séjour variable** (par type de séjour, service ou GHM) pourrait améliorer la prévision d’occupation.
- **Variables météorologiques** : Bouteloup intègre la météo (11 variables) ; l’effet est modéré mais présent (R², Pearson). Nous pourrions **ajouter température, précipitations** si des données sont disponibles (open data Météo France, par exemple) pour affiner les pics (canicule, vague de froid).
- **Jours fériés et vacances scolaires** : Bouteloup les inclut explicitement ; notre génération de données fictives peut les intégrer en dur ; les modèles de prédiction pourraient recevoir des indicatrices **jour férié, vacances scolaires (zone)** pour mieux coller à la littérature et au terrain.
- **Comparaison de modèles** : les deux thèses **comparent** plusieurs modèles (GAM vs calendrier seul vs météo ; NN vs RF vs régression). Nous avons déjà Holt-Winters, régression, SARIMA, MA ; nous pourrions documenter une **comparaison systématique** (ex. MAE, RMSE, % de jours dans une bande ±10 %) sur une période de validation fictive, pour justifier le choix du modèle dans le rapport de conception.

#### Validation et métriques
- **Critère opérationnel** : Bouteloup utilise « % de jours à ±10 % ». Nous pouvons **définir un critère similaire** sur nos prévisions (ex. % de jours où la prédiction d’admissions ou d’occupation est dans une bande ±10 % ou dans l’IC 95 %) et le rapporter dans le rapport et, si possible, dans le dashboard.
- **Asymétrie des erreurs** : la sous-estimation du flux est plus risquée pour l’hôpital (sous-dimensionnement). Nous pouvons **analyser et rapporter** si nos modèles sous-estiment ou surestiment en moyenne, et **ajuster les recommandations** (ex. privilégier la borne haute de l’IC pour la planification).

#### Périmètre fonctionnel
- **Hospitalisations non programmées** : Bouteloup souligne que prédire les **hospitalisations non programmées** (et pas seulement les passages) serait plus utile pour les lits d’aval. Notre prototype prédit admissions et occupation ; nous pouvons **clarifier** dans le rapport que, avec des données réelles, une cible pertinente serait aussi les **entrées en hospitalisation depuis les urgences** (flux d’aval).
- **Adoption et impact** : Batal et al. montrent un **impact mesuré** (baisse des départs sans soins, des plaintes) quand le planning est adapté à la prédiction. Dans le rapport stratégique, nous pouvons citer ces résultats pour étayer l’**évaluation d’impact potentiel** de notre outil (réduction des temps d’attente, meilleure répartition des ressources).

---

## 4. Constats à intégrer dans les livrables

### 4.1 Rapport de conception et d’analyse hospitalière
- **État de l’art** : citer les deux thèses — Lequertier pour la prédiction de la durée de séjour et les données PMSI ; Bouteloup pour la prédiction du flux journalier aux urgences (GAM, calendrier, lags, météo) et le critère ±10 %.
- **Choix des modèles** : justifier Holt-Winters et régression (lags + calendrier) au regard de la littérature (Bouteloup, séries temporelles, GAM) ; mentionner la possibilité d’ajouter météo et jours fériés / vacances.
- **Limites** : préciser que les données sont **fictives** et que la validation opérationnelle nécessiterait des données réelles et un cadre réglementaire ; discuter l’utilisation d’une **durée de séjour moyenne** vs une durée prédite par segment (réf. Lequertier).
- **Validation** : si possible, rapporter une métrique de type « % de jours à ±10 % » sur une période de test fictive et commenter l’asymétrie des erreurs (sous- vs surestimation).

### 4.2 Rapport stratégique
- **Freins et leviers** : évoquer la **nécessité de données réelles** (PMSI, RPU) et d’autorisations (CEREES, CNIL, hébergement HDS) pour un déploiement en routine, en s’appuyant sur le cadre décrit par Lequertier.
- **Impact potentiel** : s’appuyer sur **Batal et al.** (réduction des départs sans soins et des plaintes lorsque le planning est adapté à la prédiction) pour argumenter l’intérêt d’un outil de prévision du flux et des besoins en lits/personnel.
- **Comparaison avec l’existant** : mentionner les travaux sur le flux aux urgences (GAM, ARIMA, réseaux de neurones) et sur la durée de séjour (NN, RF, régression logistique) pour situer notre MVP (flux agrégé + occupation + scénarios) dans le paysage des solutions.

### 4.3 Prototype et dashboard
- **Recommandations** : intégrer l’idée que la **sous-estimation** du flux est plus risquée ; afficher ou souligner la **borne haute** de l’IC (ou un scénario prudent) pour la planification des ressources.
- **Évolutions possibles** : documenter les pistes suivantes — (1) intégration de **données météo** ; (2) indicatrices **jours fériés / vacances scolaires** ; (3) **durée de séjour** variable ou prédite (si données PMSI disponibles) ; (4) indicateur cible **hospitalisations non programmées** en plus des admissions brutes.

---

## 5. Références bibliographiques (à inclure dans les rapports)

- **Lequertier V.** Méthode globale de prédiction des durées de séjours hospitalières avec intégration des données incrémentales et évolutives. Thèse de doctorat, Université Claude Bernard Lyon 1, 2022. HAL tel-04053390.
- **Bouteloup F.** Création et validation d’un modèle de prédiction du nombre de passages journaliers dans le service des urgences de l’Hôpital Pellegrin. Thèse pour le diplôme d’État de docteur en médecine, Université de Bordeaux, 2020. HAL dumas-03085214.
- **Batal H. et al.** Predicting patient visits to an urgent care clinic using calendar variables. Acad Emerg Med. 2001;8(1):48-53. (adaptation du planning, réduction des départs sans soins et des plaintes.)

---

---

## 6. Améliorations implémentées dans le code (post-analyse)

Les améliorations suivantes, tirées des thèses, ont été **implémentées dans le prototype** :

| Piste (thèses) | Implémentation |
|----------------|-----------------|
| **Jours fériés et vacances scolaires** (Bouteloup) | Module `src/prediction/calendar_utils.py` : jours fériés français (fixes + Pâques), veille/lendemain de férié, vacances scolaires zone C (Paris). Intégrés comme variables dans la régression. |
| **Lags 7–13** (Bouteloup) | Régression : ajout de la feature `lag_mean_7_13` (moyenne des valeurs j-7 à j-13). |
| **Variables météorologiques** (Bouteloup) | Température synthétique (courbe sinusoïdale Paris) ajoutée comme variable optionnelle dans la régression (structure prête pour données Météo France). |
| **Durée de séjour variable** (Lequertier) | `_duree_sejour_saisonniere(month)` : hiver (11–2) +8 %, été (6–8) −8 %. Utilisée dans `predict_occupation_from_admissions` (option `duree_sejour_saisonniere=True`). |
| **Critère de validation ±10 %** (Bouteloup) | Fonction `evaluate_forecast_pct_within_10(series, validation_days=90)` : % de jours à ±10 %, biais moyen, % surestimation/sous-estimation. Affichée dans le tableau de bord (section Prévisions, expander). |

*Document rédigé pour le projet Data Pitié-Salpêtrière (Promo 2026). Les constats des thèses sont intégrés de manière critique dans les livrables et le prototype.*
