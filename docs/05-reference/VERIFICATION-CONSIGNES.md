# Vérification — Réponse aux consignes (Consignes projet Data-1.pdf)

Ce document confronte **chaque demande explicite du PDF** au contenu actuel du projet.  
Légende : **OK** = couvert · **Partiel** = partiellement couvert · **À finaliser** = manquant ou à rédiger/déposer.

---

## 1. Introduction et objectifs (PDF § 1.1)

| Consigne | Statut | Emplacement / remarque |
|----------|--------|-------------------------|
| Générer ou trouver un jeu de données **fictif** basé sur l'activité de la **Pitié-Salpêtrière** | **OK** | `src/data/generate.py`, `data/generated/` — tendances réalistes (saisonnalité, services, jour semaine) |
| Analyser l'**évolution des admissions** et l'**impact sur les ressources** disponibles | **OK** | Dashboard « Flux & historique », notebooks AED, prévisions occupation |
| Développer un **modèle de prévision** pour **anticiper les pics d'activité** | **OK** | `src/prediction/` (Holt-Winters, Ridge, SARIMA, Boosting, modèle stock), dashboard « Prévisions » |
| Contexte : urgences, cardiologie, neurologie, maladies infectieuses… pics saisonniers, événements exceptionnels | **OK** | Génération (services, saisonnalité hiver), scénarios (épidémie, canicule, afflux massif) |

---

## 2. Dimensions du projet (PDF § 1.2)

| Consigne | Statut | Emplacement / remarque |
|----------|--------|-------------------------|
| **Génération** d'un jeu de données fictif (Pitié-Salpêtrière) | **OK** | `src/data/generate.py`, `data/generated/` |
| **Développement technique** : MVP, **tableau de bord interactif**, **module de recommandations** automatiques | **OK** | `app/dashboard.py`, `src/dashboard/`, `src/recommendations/engine.py` |
| **Conformité** réglementaire, légale et éthique (données de santé) | **OK** | `CONFORMITE.md` ; à rappeler dans le rapport de conception |
| **Analyse fonctionnelle** : besoins hospitaliers, comparaison avec pratiques actuelles | **Partiel** | Structure dans `docs/01-rapport-conception/` ; contenu à rédiger dans le **rapport de conception** |
| **Approche analytique/statistique et prédictive** : modélisations, dataviz, AED, identifier/anticiper variations et besoins | **OK** | `notebooks/01-aed/`, `src/analysis/`, `src/prediction/`, dataviz dans le dashboard |
| **Tests de scénario** : épidémie, grève, pics saisonniers… | **OK** | `src/simulation/scenarios.py`, `config/constants.py` (SCENARIOS), dashboard « Simulation de scénarios » (épidémie grippe, grève, canicule, afflux massif) |
| **Analyse stratégique et marketing** : plan d'adoption | **Partiel** | Structure dans `docs/02-rapport-strategique/` ; contenu à rédiger dans le **rapport stratégique** |
| **Étude d'impact et recommandations stratégiques** : efficacité, comparaison solutions existantes, évolutions futures | **Partiel** | Idem rapport stratégique ; pistes dans `PISTES-EVOLUTION.md` |

---

## 3. Livrables attendus (PDF § 1.3)

### 3.1 Rapport de conception et d'analyse hospitalière

| Consigne | Statut | Emplacement / remarque |
|----------|--------|-------------------------|
| **Présentation des fonctionnalités du prototype** et **méthodologie de développement** | **À finaliser** | Structure dans `docs/01-rapport-conception/README.md` ; **rédiger** `rapport-conception.md` (ou .docx/.pdf) avec ces sections |
| **Analyse approfondie des tendances d'admissions**, périodes critiques, évaluation des stratégies hospitalières actuelles | **À finaliser** | À rédiger dans le rapport de conception (s'appuyer sur AED, littérature, dashboard) |
| **Présentation de l'analyse statistique** : justifications des **dataviz** implémentées, **modèles statistiques** utilisés, évaluation de leur applicabilité | **Partiel** | Justification modèles : `JUSTIFICATION-MODELES-PREDICTION.md` ; dataviz et applicabilité à intégrer dans le **rapport** |
| **Présentation du/des modèles de prédiction** entraînés, **justification des choix**, **évaluation de l'impact** de leur utilisation | **Partiel** | Justification détaillée dans `JUSTIFICATION-MODELES-PREDICTION.md` ; synthèse et impact à mettre dans le **rapport** |
| **Rappel** : définition modèle statistique (distributions, relations, paramètres ; expliquer, prédire, estimer avec incertitude) | **OK** | Déjà dans `docs/01-rapport-conception/README.md` ; à recopier ou référencer dans le rapport rédigé |

### 3.2 Prototype fonctionnel

| Consigne | Statut | Emplacement / remarque |
|----------|--------|-------------------------|
| **Interface** pour **explorer les flux hospitaliers** et **simuler différents scénarios** (ex. épidémie, grève, afflux massif) | **OK** | Dashboard : « Flux & historique », « Simulation de scénarios » avec choix (épidémie grippe, grève, canicule, afflux massif) |
| **Modélisation des tendances d'admissions** et **prévision des besoins en lits**, **personnel** et **matériel médical** | **Partiel** | **Lits** : OK (prévision occupation, taux, seuils). **Personnel / matériel** : présents dans les **recommandations** (alertes, actions : renforts, stocks) ; pas de **prévision quantitative dédiée** (ex. ETP ou quantité matériel) — acceptable en MVP si explicité |
| **Tableau de bord interactif** pour que les décideurs **ajustent les ressources** en fonction des **prévisions** | **OK** | Dashboard : prévisions (taux, IC, alertes), recommandations, simulation ; filtres et paramètres (horizon, scénario) |

### 3.3 Rapport stratégique

| Consigne | Statut | Emplacement / remarque |
|----------|--------|-------------------------|
| **Freins et leviers d'adoption** d'un outil prédictif en milieu hospitalier | **À finaliser** | Structure dans `docs/02-rapport-strategique/README.md` ; **rédiger** `rapport-strategique.md` (ou .docx/.pdf) |
| **Recommandations** pour l'amélioration de la **perception** et de l'**efficacité** des services hospitaliers | **À finaliser** | À rédiger dans le rapport stratégique |
| **Évaluation de l'impact potentiel** (réduction temps d'attente, optimisation coûts, répartition des ressources) | **À finaliser** | À rédiger ; s'appuyer sur littérature (Batal et al., Bouteloup) et résultats du prototype |
| **Comparaison avec les solutions existantes**, identification des **axes d'amélioration** | **À finaliser** | À rédiger dans le rapport stratégique |
| **Version complète** du prototype et des simulations, **explication des choix techniques et analytiques** | **Partiel** | Choix techniques/analytiques : `JUSTIFICATION-MODELES-PREDICTION.md`, `VUE-ENSEMBLE-PROJET.md` ; à synthétiser dans le rapport stratégique |
| **Analyse des résultats du modèle prédictif** et **recommandations concrètes** pour améliorer la gestion hospitalière | **Partiel** | Résultats dans dashboard et `03-modeles-et-resultats/` ; recommandations automatiques dans `src/recommendations/` ; à commenter dans le rapport |
| **Discussion** sur les **perspectives d'évolution** de l'outil et **applications futures** | **OK** | `docs/05-reference/PISTES-EVOLUTION.md` ; à intégrer ou référencer dans le rapport stratégique |

---

## 4. Soutenance (PDF § 1.4)

| Consigne | Statut | Emplacement / remarque |
|----------|--------|-------------------------|
| **20 minutes** de présentation + **5 à 10 minutes** de questions | **OK** | `docs/soutenance/README.md` (timing indicatif) |
| **Direction de l'hôpital Pitié-Salpêtrière** = cible (ton et contenu) | **OK** | Rappels dans README soutenance et VUE-ENSEMBLE-PROJET |
| **Tous les membres** du groupe participent ; absence = échec | **OK** | Rappel dans README soutenance ; à gérer en préparation (répartition des rôles) |
| **5 min** présentation **commerciale** : contexte, analyse des besoins, modèles, conclusion représentative | **OK** | Plan dans `docs/soutenance/README.md` |
| **10 min** présentation **technique** : échanges avec le jury (aspects techniques, fonctionnels, stratégiques) | **OK** | Plan dans `docs/soutenance/README.md` |
| **Support** de présentation (PDF ou lien) | **À finaliser** | À déposer dans `docs/soutenance/` ; un fichier `.pptx` existe à la racine — à déplacer ou lier et vérifier qu’il couvre le plan |

---

## 5. Ressources (PDF § 1.5)

| Consigne | Statut | Emplacement / remarque |
|----------|--------|-------------------------|
| INSEE, DREES, Géodes, data.gouv.fr (données santé) | **OK** | Liens dans `README.md` (racine du projet) |

---

## Synthèse : points à finaliser pour être 100 % aligné

1. **Rédiger le rapport de conception** (`rapport-conception.md` ou .docx/.pdf) en suivant la structure de `docs/01-rapport-conception/README.md`, en intégrant fonctionnalités, méthodologie, tendances, analyse statistique, modèles (avec renvoi à `JUSTIFICATION-MODELES-PREDICTION.md`) et évaluation de l’impact.
2. **Rédiger le rapport stratégique** (`rapport-strategique.md` ou .docx/.pdf) en suivant `docs/02-rapport-strategique/README.md` : freins/leviers, recommandations, impact, comparaison avec l’existant, perspectives (avec renvoi à `PISTES-EVOLUTION.md`).
3. **Soutenance** : déposer le support final (PDF ou lien) dans `docs/soutenance/` et s’assurer qu’il suit le plan (5 min commercial, 10 min technique, démo).
4. **Optionnel** : dans le rapport de conception, préciser que la prévision quantitative « personnel » et « matériel médical » est abordée via les **recommandations** (alertes et actions) à partir des prévisions de **lits** ; une évolution possible serait des modules dédiés (ratios ETP, stocks).

---

*Référentiel détaillé : [REFERENTIEL-CONSIGNES.md](REFERENTIEL-CONSIGNES.md).*
