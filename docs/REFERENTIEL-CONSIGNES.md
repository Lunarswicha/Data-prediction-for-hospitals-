# Référentiel — Suivi précis des consignes

Ce document permet de vérifier que chaque point des consignes officielles est couvert.

---

## 1. Dimensions du projet (PDF § 1.2)

| Consigne | Couverture |
|----------|------------|
| Génération d’un jeu de données fictif basé sur l’activité de la Pitié-Salpêtrière | `data/generated/` · `src/data/` |
| Développement technique : MVP, tableau de bord interactif, module de recommandations | `src/` · `app/` |
| Conformité réglementaire et éthique (données de santé) | `CONFORMITE.md` · rapport conception |
| Analyse fonctionnelle : besoins hospitaliers, comparaison avec pratiques actuelles | `docs/01-rapport-conception/` — section dédiée |
| Approche analytique/statistique et prédictive : modélisations, dataviz, AED | `notebooks/` · `src/analysis/` · `src/prediction/` |
| Tests de scénario (épidémie, grève, pics saisonniers…) | `src/simulation/` · rapport |
| Analyse stratégique et marketing : plan d’adoption | `docs/02-rapport-strategique/` |
| Étude d’impact et recommandations stratégiques | `docs/02-rapport-strategique/` |

---

## 2. Livrables attendus (PDF § 1.3)

### Rapport de conception et d’analyse hospitalière

| Point consigne | Où le traiter |
|----------------|---------------|
| Présentation des fonctionnalités du prototype et méthodologie de développement | Rapport conception § 2 |
| Analyse approfondie des tendances d’admissions, périodes critiques, stratégies actuelles | Rapport conception § 3 |
| Présentation de l’analyse statistique, justifications des dataviz, modèles statistiques, applicabilité | Rapport conception § 4 |
| Présentation du/des modèles de prédiction, justification des choix, évaluation de l’impact | Rapport conception § 5 |
| Rappel : définition modèle statistique (distributions, relations, paramètres ; expliquer, prédire, estimer avec incertitude) | Rapport conception § 4–5 |

### Prototype fonctionnel

| Point consigne | Où le traiter |
|----------------|---------------|
| Interface pour explorer les flux hospitaliers et simuler différents scénarios (épidémie, grève, afflux massif) | `app/` · `src/simulation/` |
| Modélisation des tendances d’admissions et prévision des besoins (lits, personnel, matériel médical) | `src/prediction/` · `src/analysis/` |
| Tableau de bord interactif pour ajuster les ressources en fonction des prévisions | `src/dashboard/` · `app/` |

### Rapport stratégique

| Point consigne | Où le traiter |
|----------------|---------------|
| Freins et leviers d’adoption d’un outil prédictif en milieu hospitalier | Rapport stratégique § 2 |
| Recommandations pour amélioration perception et efficacité des services | Rapport stratégique § 3 |
| Évaluation de l’impact potentiel (temps d’attente, coûts, répartition des ressources) | Rapport stratégique § 4 |
| Comparaison avec les solutions existantes, axes d’amélioration | Rapport stratégique § 5 |
| Explication des choix techniques et analytiques, analyse des résultats du modèle prédictif | Rapport stratégique § 3–4 |
| Perspectives d’évolution de l’outil et applications futures | Rapport stratégique § 6 |

---

## 3. Soutenance (PDF § 1.4)

| Consigne | Application |
|----------|-------------|
| 20 min présentation + 5–10 min questions | `docs/soutenance/` — timing |
| Direction Pitié-Salpêtrière = cible | Ton et contenu orientés décideurs |
| Tous les membres participent ; absence = échec | Répartition des rôles dans le plan |
| 5 min présentation commerciale (contexte, besoins, modèles, conclusion) | Plan soutenance |
| 10 min présentation technique (échanges jury) | Plan soutenance |

---

## 4. Ressources (PDF § 1.5)

- INSEE, DREES, Géodes, data.gouv.fr — liens dans `README.md`.

---

*Document de suivi à cocher au fur et à mesure de l’avancement.*
