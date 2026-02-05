# Projet Data — Pitié-Salpêtrière

**Promo 2026** · MVP de simulation et prévision des besoins hospitaliers

---

## Vue d'ensemble

L’Hôpital Pitié-Salpêtrière (Paris) doit anticiper les pics d’admission et optimiser ses ressources. Ce dépôt contient l’architecture du projet, le prototype fonctionnel et les livrables documentaires, **alignés sur les consignes officielles**.

---

## Architecture du projet

```
Data Hopital/
├── README.md                      (vous êtes ici)
├── CONFORMITE.md                  Conformité données de santé (RGPD, éthique)
│
├── docs/                          DOCUMENTATION (voir docs/README.md)
│   ├── README.md                  Index par catégories
│   ├── VUE-ENSEMBLE-PROJET.md     Explication globale, rapports, chiffres, modèles
│   ├── 01-rapport-conception/     Rapport de conception et d'analyse hospitalière
│   ├── 02-rapport-strategique/    Rapport stratégique
│   ├── 03-modeles-et-resultats/   Modèles, cohérence, explication des %
│   ├── 04-litterature/            Thèses (Lequertier, Bouteloup), veille
│   ├── 05-reference/              Consignes, pistes d'évolution
│   └── soutenance/                Plan et support soutenance (20 min)
│
├── data/                          DONNÉES
│   ├── raw/                       Données brutes (sources, open data)
│   ├── generated/                 Jeu de données fictif Pitié-Salpêtrière
│   └── processed/                 Données transformées pour le prototype
│
├── notebooks/                     ANALYSES & EXPLORATION
│   ├── 01-aed/                    Analyse exploratoire des données
│   ├── 02-modeles-statistiques/   Modélisation statistique
│   └── 03-modeles-prediction/     Entraînement et évaluation des modèles
│
├── src/                           CODE DU PROTOTYPE (MVP)
│   ├── data/                      Génération / chargement des données
│   ├── analysis/                  AED, modèles statistiques, dataviz
│   ├── prediction/                Modèles de prédiction (pics, besoins)
│   ├── simulation/                Simulation de scénarios (épidémie, grève, etc.)
│   ├── dashboard/                 Tableau de bord interactif
│   └── recommendations/          Module de recommandations automatiques
│
├── app/                           APPLICATION (interface utilisateur)
│   └── (Streamlit / Dash / etc.)  Point d'entrée du tableau de bord
│
└── config/                        CONFIGURATION
    └── (paramètres, constantes, scénarios)
```

---

## Mapping consignes → livrables

| Consigne (PDF) | Emplacement / livrable |
|----------------|------------------------|
| **Génération jeu de données fictif** (Pitié-Salpêtrière) | `data/generated/` · `src/data/` |
| **Développement technique** · MVP · tableau de bord · recommandations | `src/` · `app/` |
| **Conformité réglementaire & éthique** (données de santé) | `CONFORMITE.md` · conception dans rapport |
| **Analyse fonctionnelle** (besoins, pratiques actuelles) | `docs/01-rapport-conception/` — section Analyse fonctionnelle |
| **Approche analytique/statistique et prédictive** · AED · dataviz · modèles | `notebooks/` · `src/analysis/` · `src/prediction/` |
| **Tests de scénarios** (épidémie, grève, pics saisonniers) | `src/simulation/` · rapport conception |
| **Rapport de conception et d'analyse hospitalière** | `docs/01-rapport-conception/` |
| **Prototype fonctionnel** (interface, simulation, prévisions, dashboard) | `src/` · `app/` |
| **Rapport stratégique** (freins/leviers, impact, comparaison, évolutions) | `docs/02-rapport-strategique/` |
| **Soutenance** (20 min, direction hôpital) | `docs/soutenance/` |

---

## Livrables attendus (2 rapports + 1 prototype)

1. **Document 1** — Rapport de conception et d'analyse hospitalière **+** Étude d'impact et recommandations stratégiques **= 1 document**  
   → `docs/01-rapport-conception/`

2. **Document 2** — Rapport stratégique et marketing **= 1 document**  
   → `docs/02-rapport-strategique/`

3. **Prototype** — Simple, efficace (tableau de bord, prévisions, simulation, recommandations)  
   → `app/` · `src/`v.fr)
- [Santé publique France – Géodes](https://geodes.santepubliquefrance.fr)
- [data.gouv.fr – Données santé](https://www.data.gouv.fr/fr/pages/donnees_sante/)

---

## Démarrage rapide

```bash
# Environnement
python -m venv venv
source venv/bin/activate   # Windows : venv\Scripts\activate
pip install -r requirements.txt

# 1. Générer les données fictives (une fois)
python -m src.data.generate

# 2. Lancer le tableau de bord
streamlit run app/dashboard.py
```

Ouvrir http://localhost:8501 dans le navigateur.  
Sections : **Flux & historique** · **Prévisions** · **Simulation de scénarios** · **Recommandations**.

---

*Projet aligné sur les consignes projet DATA — Promo 2026.*
