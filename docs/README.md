# Documentation du projet — Pitié-Salpêtrière
---

## 🚨 URGENT — CORRECTIONS DU 5 FÉVRIER 2026

**Statut : 🟢 BUG CRITIQUE CORRIGÉ — PROJET PRÊT POUR SOUTENANCE**

### 📖 Démarrage rapide (lisez dans cet ordre) :
1. **[CHECKLIST-SOUTENANCE.md](CHECKLIST-SOUTENANCE.md)** ← **COMMENCER ICI** (5 min avant la soutenance)
2. **[RESUME-FINAL.md](RESUME-FINAL.md)** ← Rapport complet de la correction (lecture : 10 min)
3. **[DIAGNOSTIC-PREDICTIONS.md](DIAGNOSTIC-PREDICTIONS.md)** ← Analyse détaillée du bug (optionnel)
4. **[CORRECTION-VALIDEE.md](CORRECTION-VALIDEE.md)** ← Détails techniques de la correction (optionnel)

### ⚡ Résumé ultra-rapide :
- ✅ **Bug identifié** : Le modèle stock-flux utilisait une formule statique défectueuse
- ✅ **Correction** : Modèle dynamique jour par jour (physiquement cohérent)
- ✅ **Validation** : 92% des jours avec variations dynamiques (au lieu de <20%)
- ✅ **Dashboard** : Testé, fonctionne correctement, aucune régression
- ✅ **Statut** : **PRÊT POUR LA SOUTENANCE**

---

## 📚 Documentation du projet
Index de la documentation, **organisée par catégories**. Pour une vue d’ensemble du projet, des rapports et des résultats liés aux chiffres et modèles, voir **[VUE-ENSEMBLE-PROJET.md](VUE-ENSEMBLE-PROJET.md)**.

---

## Livrables demandés (2 rapports + 1 prototype)

| Livrable | Contenu |
|----------|---------|
| **Document 1** | Rapport de conception et d'analyse hospitalière **+** Étude d'impact et recommandations stratégiques **= 1 document** → [01-rapport-conception/](01-rapport-conception/) |
| **Document 2** | Rapport stratégique et marketing **= 1 document** → [02-rapport-strategique/](02-rapport-strategique/) |
| **Prototype** | Simple, efficace (tableau de bord, prévisions, simulation, recommandations) → `app/` · `src/` |

---

## 1. Livrables (rapports et soutenance)

| Dossier | Contenu |
|---------|---------|
| **[01-rapport-conception/](01-rapport-conception/)** | Rapport de conception et d’analyse hospitalière : structure recommandée, fonctionnalités du prototype, tendances, analyse statistique, modèles de prédiction, justification (JUSTIFICATION-MODELES-PREDICTION.md). |
| **[02-rapport-strategique/](02-rapport-strategique/)** | Rapport stratégique : freins/leviers d’adoption, recommandations, impact, comparaison avec l’existant, perspectives. |
| **[soutenance/](soutenance/)** | Plan et support pour la soutenance (20 min, direction Pitié-Salpêtrière). |

---

## 2. Modèles et résultats

| Dossier | Contenu |
|---------|---------|
| **[03-modeles-et-resultats/](03-modeles-et-resultats/)** | Cohérence mathématique des modèles (admissions, occupation, besoins), formules, contradictions et complémentarités ; **explication des pourcentages** affichés (taux d’occupation, % à ±10 % Bouteloup, seuils 85 %/95 %). |

**Documents :** [COHERENCE-MODELES-PREVISION.md](03-modeles-et-resultats/COHERENCE-MODELES-PREVISION.md), [EXPLICATION-POURCENTAGES.md](03-modeles-et-resultats/EXPLICATION-POURCENTAGES.md).

---

## 3. Littérature et veille

| Dossier | Contenu |
|---------|---------|
| **[04-litterature/](04-litterature/)** | Analyse critique des thèses intégrées (Lequertier 2022, Bouteloup 2020) ; veille thèses et doctorats (Wargon, études multicentriques, Île-de-France) ; améliorations du modèle (splines, lags, critère ±10 %). |

**Documents :** [ANALYSE-CRITIQUE-LITTERATURE-THESES.md](04-litterature/ANALYSE-CRITIQUE-LITTERATURE-THESES.md), [VEILLE-THESES-DOCTORATS.md](04-litterature/VEILLE-THESES-DOCTORATS.md).

---

## 4. Référence et évolution

| Dossier | Contenu |
|---------|---------|
| **[05-reference/](05-reference/)** | Référentiel de suivi des consignes (mapping PDF → livrables) ; pistes d’évolution du MVP (modèles, scénarios, alertes, données, interface). |

**Documents :** [REFERENTIEL-CONSIGNES.md](05-reference/REFERENTIEL-CONSIGNES.md), [PISTES-EVOLUTION.md](05-reference/PISTES-EVOLUTION.md).

---

## Arborescence

```
docs/
├── README.md                    (ce fichier — index par catégories)
├── VUE-ENSEMBLE-PROJET.md       (explication globale, rapports, chiffres, modèles)
├── 01-rapport-conception/       Livrables : conception
├── 02-rapport-strategique/      Livrables : stratégique
├── 03-modeles-et-resultats/     Modèles, cohérence, % affichés
├── 04-litterature/              Thèses, veille, améliorations
├── 05-reference/                Consignes, pistes d’évolution
└── soutenance/                  Soutenance
```

---

*Projet Data Promo 2026 — Données fictives, pas de données réelles.*
