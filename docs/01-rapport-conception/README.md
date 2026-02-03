# Livrable 1 — Rapport de conception et d'analyse hospitalière

*Consignes : présentation des fonctionnalités du prototype, méthodologie, analyse des tendances, analyse statistique, modèles, évaluation de l’impact.*

**Littérature** : voir `../ANALYSE-CRITIQUE-LITTERATURE-THESES.md` (thèses Lequertier, Bouteloup) pour constats à intégrer dans ce rapport.

**Justification des modèles (niveau M2)** : voir `JUSTIFICATION-MODELES-PREDICTION.md` pour les choix méthodologiques, hypothèses, limites et protocole de validation des modèles de prédiction.

---

## Structure recommandée du rapport

1. **Introduction & objectifs**
   - Contexte Pitié-Salpêtrière, objectifs du MVP.

2. **Fonctionnalités du prototype & méthodologie de développement**
   - Description des modules (données, analyse, prédiction, simulation, dashboard, recommandations).
   - Stack technique, choix d’architecture, conformité (réf. `CONFORMITE.md`).

3. **Analyse des tendances d’admissions**
   - Périodes critiques, stratégies hospitalières actuelles (benchmark / littérature).

4. **Analyse statistique & dataviz**
   - Justification des visualisations implémentées.
   - Modèles statistiques utilisés et applicabilité.

5. **Modèles de prédiction**
   - **Justification détaillée** : familles envisagées (séries temporelles, régression, ML), choix retenus (Holt-Winters, régression Ridge, SARIMA, modèle stock), hypothèses et limites (réf. `JUSTIFICATION-MODELES-PREDICTION.md`).
   - Évaluation (métrique ±10 %, biais, IC 95 %) et impact de leur utilisation.

6. **Synthèse & perspectives**
   - Synthèse des résultats, limites, pistes d’amélioration.

---

## Rappel consigne — Modèle statistique

*Extrait consignes : « En data science, un modèle statistique est une représentation mathématique (généralement basée sur la théorie des probabilités) qui décrit la manière dont sont générées les données observées » — hypothèses sur distributions, relations entre variables, paramètres ; objectifs : expliquer, prédire, estimer avec incertitude.*

---

À rédiger dans ce dossier :  
- `rapport-conception.md` (ou .docx / .pdf exporté)  
- Figures, dataviz exportées depuis les notebooks / le prototype.
