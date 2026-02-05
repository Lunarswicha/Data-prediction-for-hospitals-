# Cohérence mathématique des modèles prévisionnels

Ce document vérifie la cohérence des modèles de prédiction (admissions, occupation, besoins), signale les éventuelles contradictions et les complémentarités, et indique dans quels cas les estimations se rapprochent ou divergent.

---

## 1. Inventaire des modèles et formules

### 1.1 Prédiction des **admissions** (volume journalier)

| Modèle | Formule / hypothèses | Sortie | IC 95 % |
|--------|----------------------|--------|---------|
| **Moyenne glissante (MA)** | \( \hat{y}_{t+h} = \bar{y}_{[t-w:t]} + \tau \cdot (h+1) \), \( \tau = \frac{\bar{y}_{t-7:t} - \bar{y}_{t-w:t-7}}{7} \) | admissions/jour | \( \hat{y} \pm 1{,}96 \, \sigma_{res} \) (σ sur fenêtre) |
| **Holt-Winters** | Lissage exponentiel additif, saisonnalité période 7, tendance additive | admissions/jour | \( \hat{y} \pm 1{,}96 \, \sigma_{residus} \) |
| **Régression Ridge + splines** | \( \hat{y} = \beta^T x \), \( x = \) lags (1,7,14, mean 7–13), calendrier, splines(jour_semaine, jour_du_mois, temp.) | admissions/jour | \( \hat{y} \pm 1{,}96 \, \sigma_{residus} \) |
| **SARIMA** | ARIMA avec composante saisonnière (7 j) | admissions/jour | conf_int(0,05) ou \( \hat{y} \pm 1{,}96\sqrt{\sigma^2} \) |
| **Boosting (XGBoost/GBM)** | Même \( x \) que Ridge + splines, modèle gradient boosting | admissions/jour | \( \hat{y} \pm 1{,}96 \, \sigma_{residus} \) |

**Choix effectif (admissions)** : `predict_admissions_best` utilise par défaut **Holt-Winters Triple Exponential Smoothing** (statsmodels) : `trend='add'`, `seasonal='add'`, `seasonal_periods=7`, IC à partir des résidus (30 derniers jours). Autres options : `best_by_backtest` (benchmark HW / Ridge / Boosting / SARIMA), ou cascade HW → Ridge → Boosting → SARIMA → MA.

---

### 1.2 Prédiction de l'**occupation** (nombre de lits, taux)

Deux **chaînes** possibles, utilisées de manière **exclusive** (une seule à la fois) :

**Chaîne A — Prédiction directe** (`predict_occupation_direct`)  
- Cible : série **occupation_lits** (nombre de lits occupés).  
- Modèles testés dans l'ordre : Holt-Winters(occupation) → Ridge(occupation) → MA + tendance.  
- Sortie : **occupation_lits_pred**, **occupation_lits_low/high**, **admissions_pred** = NaN.

**Chaîne B — Via admissions** (`predict_occupation_from_admissions`)  
- Étape 1 : prédiction des **admissions** (Holt-Winters ou Ridge sur `admissions_jour`, ou MA).  
- Étape 2 : **modèle stock simplifié** (heuristique) :
  \[
  \text{occ\_pred} = 0{,}85 \cdot \overline{\text{occ}}_{-28} + 0{,}15 \cdot \widehat{\text{adm}} \cdot \frac{\overline{\text{occ}}_{-28}}{\overline{\text{adm}}_{-28}} \cdot \text{durée}(m),
  \]
  où \( \text{durée}(m) \) = durée de séjour moyenne (saisonnière, réf. Lequertier : hiver +8 %, été −8 %).  
- Sortie : **occupation_lits_pred**, intervalles, **admissions_pred** = valeur prédite.

**Choix effectif (occupation)** : `predict_occupation_best` utilise **Chaîne A** si la prédiction directe est disponible et complète, **sinon Chaîne B**. Une seule chaîne est donc affichée à la fois.

---

### 1.3 Agrégation finale : **besoins** (`predict_besoins`)

- Entrée : sortie de `predict_occupation_best` (donc soit directe, soit via admissions).  
- Taux : `taux_occupation_pred = occupation_lits_pred / capacite_lits`.  
- Alertes : seuils 85 % (alerte) et 95 % (critique) appliqués sur ce taux.  
- Aucun autre modèle n'est mélangé ici.

---

## 2. Vérifications de cohérence mathématique

### 2.1 Unités et grandeurs

- **Admissions** : tous les modèles d'admissions prédisent un **même type de grandeur** (volume journalier, même unité). Cohérent.
- **Occupation** : prédiction en **nombre de lits** puis conversion en **taux** (0–1) par division par la capacité. Cohérent.
- **Intervalles** : partout on utilise une approche **symétrique** type \( \hat{y} \pm 1{,}96\,\sigma \) (ou conf_int pour SARIMA), interprétable comme IC 95 % sous hypothèse de normalité des résidus. Cohérent entre modèles.

### 2.2 Modèle stock (Chaîne B)

- La formule \( 0{,}85 \cdot \overline{\text{occ}} + 0{,}15 \cdot \widehat{\text{adm}} \cdot \text{ratio} \cdot \text{durée} \) est **heuristique** (poids 0,85 / 0,15 fixés), pas dérivée d'un modèle de type Little ou équation de stock explicite.  
- **Cohérent** avec l'objectif (lisser le niveau actuel et intégrer le flux entrant), mais **à documenter** comme simplification.  
- **ratio** = \( \overline{\text{occ}} / \overline{\text{adm}} \) sur les 28 derniers jours : homogène à un « équivalent durée de séjour » moyen observé. Cohérent.

### 2.3 Données d'entrée

- **Admissions** : tous les modèles d'admissions utilisent la **même** série (ex. `admissions_jour` agrégée).  
- **Occupation directe** : utilise la série **occupation_lits** (pas les admissions).  
- **Occupation via admissions** : utilise **admissions_jour** puis formule stock.  
Donc deux **sources** différentes possibles pour l'occupation (niveau vs flux), mais une seule utilisée à la fois. Pas d'incohérence de données.

---

## 3. Contradictions potentielles (deux sorties différentes pour une même question)

### 3.1 Occupation : directe vs via admissions

- **Situation** : pour un même jour et le même jeu de données, la **Chaîne A** (directe) et la **Chaîne B** (via admissions) peuvent donner des **niveaux d'occupation différents**.  
- **Raison** :  
  - A modélise directement le **niveau** (série occupation_lits).  
  - B modélise le **flux** (admissions) puis le transforme en niveau par une formule de stock simplifiée.  
- **Affichage** : une seule chaîne est utilisée (A prioritaire, B en secours). Il n'y a donc **pas de contradiction affichée** en même temps, mais **les deux méthodologies peuvent diverger** si on les compare en backtest ou en export.  
- **Recommandation** : considérer les deux comme **complémentaires** (vue niveau vs vue flux) ; en cas d'écart important, privilégier la chaîne dont la métrique de validation (ex. backtest, ±10 %) est la meilleure sur les données disponibles.

### 3.2 Admissions : modèle principal (best) vs Boosting

- **Situation** : dans l'interface, les **Prévisions** (besoins) s'appuient sur `predict_admissions_best` (HW ou Ridge selon succès), alors que l'onglet **Modèle Boosting** affiche la prévision **Boosting** (XGBoost/GBM) sur les admissions.  
- **Raison** : ce sont **deux modèles différents** (séries temporelles / régression Ridge vs arbres).  
- **Contradiction** : pas une incohérence mathématique, mais **deux estimations possibles** pour la même grandeur (admissions). Elles peuvent être **proches** (même ordre de grandeur, même tendance) ou **différentes** (surtout en cas de choc ou de régime différent).  
- **Recommandation** : les traiter comme **complémentaires** ; si les deux sont proches, renforce la confiance ; si elles divergent, utiliser le backtest / ±10 % pour guider le choix ou envisager un **ensemble** (moyenne) comme dans la littérature (p. ex. Île-de-France).

### 3.3 Ordre de priorité HW → Ridge → SARIMA → MA

- **Cohérence** : un seul modèle est retourné ; pas de mélange. L'ordre reflète une **préférence** (saisonnalité lissée, puis régression riche, puis SARIMA, puis baseline).  
- **Risque** : selon les données, HW peut réussir alors que Ridge serait plus précis (ou l'inverse). Pas de contradiction interne, mais **la métrique affichée** (ex. backtest) peut être plus favorable à un modèle qui n'est pas celui retenu par `predict_admissions_best`.  
- **Recommandation** : utiliser le backtest et le critère ±10 % pour discuter, dans le rapport, du choix de l'ordre de priorité ou d'un choix adaptatif.

---

## 4. Complémentarités (modèles qui se renforcent ou se complètent)

### 4.1 Point forecast + IC 95 %

- Tous les modèles fournissent une **prévision ponctuelle** et un **intervalle** (low/high).  
- **Complémentarité** : la courbe centrale donne le niveau attendu, l'IC donne une **fourchette plausible** (incertitude). Cohérent pour la décision (ex. planifier sur la borne haute).

### 4.2 Fallback MA

- **MA** est utilisée uniquement quand HW, Ridge et SARIMA échouent ou ne renvoient pas le bon nombre de jours.  
- **Complémentarité** : garantit qu'il y a **toujours** une prévision d'admissions (et, via Chaîne B, d'occupation possible). Pas de conflit avec les autres.

### 4.3 Direct (occupation) vs Via admissions

- **Direct** : bien adapté si la série **occupation_lits** est stable et prévisible.  
- **Via admissions** : utile si le **flux** est plus informatif (ex. forte variabilité des entrées).  
- **Complémentarité** : deux **stratégies** pour la même cible ; l'une est utilisée en priorité, l'autre en secours. On peut les comparer en backtest pour voir laquelle est la plus fiable sur les données disponibles.

### 4.4 Ridge vs Boosting (admissions)

- **Mêmes variables** (lags, calendrier, splines).  
- **Complémentarité** :  
  - Si les prévisions sont **proches** : renforce la robustesse.  
  - Si elles **divergent** : signale des régimes où les modèles ne sont pas d'accord (à investiguer).  
- Un **ensemble** (moyenne Ridge + Boosting) pourrait réduire la variance (piste déjà mentionnée dans 05-reference/PISTES-EVOLUTION.md).

### 4.5 Critère ±10 % (Bouteloup) et backtest

- **Complémentarité** : le **backtest** (prévision vs réel) et le **% à ±10 %** donnent une **évaluation commune** pour tous les modèles d'admissions (best, MA, Boosting). On peut ainsi comparer Ridge, HW et Boosting sur la même période et voir lesquels se **rapprochent** le plus du réel.

---

## 5. Synthèse : quand les estimations se rapprochent vs divergent

| Contexte | Attente | Commentaire |
|----------|--------|-------------|
| **HW vs Ridge (admissions)** | Souvent **proches** | Même cible, même horizon ; HW = saisonnalité lissée, Ridge = covariables explicites. En régime stable, ordres de grandeur similaires. |
| **Ridge vs Boosting (admissions)** | Souvent **proches** | Mêmes variables ; Boosting peut capturer des non-linéarités supplémentaires. En général même échelle ; écarts possibles en cas de choc ou de régime rare. |
| **Direct vs Via admissions (occupation)** | Peuvent **diverger** | Direct = niveau ; Via = flux transformé. Écarts possibles si la relation admissions–occupation n'est pas bien approchée par la formule heuristique. |
| **MA vs HW / Ridge** | Peuvent **diverger** | MA = tendance simple ; HW/Ridge = saisonnalité et calendrier. MA peut sous/surestimer en période de pic ou de creux. |
| **Prévision centrale vs IC** | Toujours **cohérentes** | La courbe centrale est au milieu de l'intervalle par construction. |

---

## 6. Recommandations pour le rapport et l'usage

1. **Documenter** le modèle stock comme **heuristique** (poids 0,85 / 0,15) et la possible **divergence** direct vs via admissions.  
2. **Interpréter** Ridge vs Boosting comme **complémentaires** ; en cas de forte divergence, utiliser le backtest et le ±10 % pour trancher ou pour justifier un futur modèle ensemble.  
3. **Ne pas mélanger** les sorties de deux chaînes (direct vs via admissions) dans un même indicateur sans le préciser.  
4. **Conserver** l'ordre HW → Ridge → SARIMA → MA comme choix par défaut, mais **évaluer** périodiquement (backtest) pour ajuster l'ordre ou le choix de modèle si des données réelles sont disponibles.

---

## 7. Simulation de scénarios : sur quoi se basent-ils ?

Les scénarios (épidémie grippe, grève, canicule, afflux massif) sont définis dans **`config/constants.py`** (`SCENARIOS`) et exécutés par **`src/simulation/scenarios.py`**. Ils restent **cohérents avec le projet** sur les **seuils et la capacité**, mais n'utilisent **pas** le même moteur que la page Prévisions.

### 7.1 Paramètres (config)

| Scénario | Paramètres principaux | Source |
|----------|------------------------|--------|
| **Épidémie grippe** | +35 % admissions, 45 j, pic au jour 15 | `config/constants.py` |
| **Grève** | −15 % capacité effective (lits), 14 j | id. |
| **Canicule** | +15 % admissions, 21 j, pic au milieu | id. |
| **Afflux massif** | +80 % admissions, 3 j, pic au jour 1 | id. |

Même **capacité** (1800 lits) et **seuils d'alerte** (85 % / 95 %) que le reste du projet → cohérent.

### 7.2 Base des scénarios (admissions)

- **Scénarios « admissions »** (épidémie, canicule, afflux massif) : la **base** est la **moyenne des 28 derniers jours** de la série d'admissions historiques (`base_series.iloc[-28:].mean()`), **pas** la prévision Holt-Winters.
- Une **courbe en cloche** (gaussienne) est appliquée pour modéliser le surplus (pic au `pic_jour`, amplitude `surplus_admissions`). Interprétation : « et si le niveau récent des admissions augmentait de X % selon cette forme de vague ».

### 7.3 Occupation dans les scénarios

- Pour les scénarios admissions, l’**occupation** est estimée par la règle **`admissions_scenario × 6`** (6 jours de séjour fixes), et le taux par division par la capacité (1800). Ce n’**est pas** le modèle stock utilisé en prévision (`predict_occupation_from_admissions` avec DMS saisonnière et inertie).
- **Scénario grève** : base = moyenne 28 j de l’**occupation** observée ; on simule une **réduction de capacité effective** (ramp-up sur 2 j) et une **augmentation d’occupation** (0,8 %/j) pour mimiquer l’accumulation. Taux = occupation / capacité effective.

### 7.4 Synthèse cohérence

| Élément | Cohérent avec le projet ? |
|--------|----------------------------|
| Capacité (1800), seuils 85 % / 95 % | **Oui** |
| Paramètres des scénarios (config) | **Oui** (référence rapport de conception, consignes) |
| Base = moyenne 28 j (au lieu de prévision HW) | **Choix volontaire** : scénario = stress-test par rapport au « niveau récent », pas à la courbe prévisionnelle |
| Occupation = admissions × 6 (au lieu du modèle stock) | **Simplification** : ordre de grandeur pour le stress-test ; pour une cohérence totale avec la prévision, on pourrait alimenter `predict_occupation_from_admissions` avec les admissions simulées du scénario |

En résumé : les scénarios sont **alignés** sur la config (seuils, capacité, types d’événements) et sur l’objectif « tests de scénario » du projet. Ils s’appuient sur une **base simple** (moyenne 28 j) et une **règle d’occupation simplifiée** (× 6) pour rester lisibles et rapides ; une évolution possible est d’utiliser la prévision HW comme base et le modèle stock pour l’occupation si on souhaite une chaîne 100 % identique à la page Prévisions.

---

*Document généré pour vérification de la cohérence des modèles prévisionnels du projet Pitié-Salpêtrière.*
