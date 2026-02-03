# Conformité réglementaire et éthique — Données de santé

*Référence consignes : développement technique « dans le respect de la conformité réglementaire, en veillant au respect des contraintes légales et éthiques liées aux données de santé ».*

---

## 1. Cadre applicable

- **RGPD** (règlement UE 2016/679) : traitement de données à caractère personnel, dont données de santé.
- **Droit français** : Code de la santé publique, Loi du 26 janvier 2016 (open data santé), recommandations CNIL.
- **Données du projet** : jeu de données **fictif** / synthétique, pas de données réelles de patients.

---

## 2. Principes retenus pour le MVP

| Principe | Application dans le projet |
|----------|----------------------------|
| **Données fictives** | Uniquement données générées ou agrégées anonymisées ; aucune donnée nominative réelle. |
| **Finalité** | Simulation et prévision des besoins (lits, personnel, matériel) pour la direction ; pas de suivi individuel. |
| **Minimisation** | Ne collecter/générer que les champs nécessaires aux modèles et au dashboard. |
| **Documentation** | Ce document + section dédiée dans le rapport de conception. |

---

## 3. À documenter dans le rapport de conception

- Choix des variables du jeu de données fictif (pourquoi ces champs, pas d’identifiants directs).
- Hypothèses d’anonymisation si agrégation de données ouvertes.
- Périmètre d’usage du prototype (interne, démo, pas de production sur données réelles sans cadre légal).

---

## 4. Évolutions en cas d’utilisation en établissement

- Analyse d’impact (AIP) si traitement de données personnelles réelles.
- DPO et registre des traitements.
- Hébergement des données de santé (HDS) si applicable.

---

*À compléter par le groupe avec les références précises et les choix effectifs du projet.*
