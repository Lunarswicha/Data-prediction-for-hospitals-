# Correction Finale des Prévisions - 5 février 2026

## 🎯 Problème Initial

L'utilisateur a identifié une **faille critique** dans les prédictions :
> "Si l'on suivait le pattern de prédictions sur 2-3 ans, à la fin il n'y aurait plus aucun patient"

Les prédictions d'admissions décroissaient continuellement de **389 → 243** sur 180 jours (-37%), et l'occupation de **1404 → 1130 lits** (-19.5%), ce qui serait **insoutenable** à long terme.

## 🔍 Analyse de la Cause

Le système utilisait `select_best_model_by_backtest()` qui sélectionnait le modèle **Ridge regression** car il avait les meilleures métriques de validation :
- MAE = 21.7 (vs 51.7 pour Holt-Winters)
- 77.8% des prédictions within ±10% (vs 30% pour HW)

**MAIS** : Ridge extrapolait une **tendance baissière récente** des données historiques, produisant des prédictions qui descendaient indéfiniment vers zéro. Ce modèle était mathématiquement précis sur le backtest, mais **physiquement irréaliste** sur long terme.

## ✅ Solution Implémentée

### 1. Nouveau Holt-Winters "Stable"

Réécriture complète de `predict_holt_winters()` pour **forcer** la stabilité :

```python
def predict_holt_winters(series, horizon_jours=14, seasonal_period=7):
    # Niveau stable = moyenne des 90 derniers jours (FIGÉ)
    window = min(90, len(series))
    niveau_stable = series.iloc[-window:].mean()
    
    # Saisonnalité = UNIQUEMENT hebdomadaire (écarts par jour de semaine)
    for dow in range(7):
        days_of_week = series_recent[series_recent.index.dayofweek == dow]
        saisonnalite_hebdo[dow] = days_of_week.mean() - niveau_stable
    
    # Prédictions = niveau_stable + saisonnalite[jour_semaine]
    for d in dates:
        pred = niveau_stable + saisonnalite_hebdo[d.dayofweek]
```

**Caractéristiques** :
- ✅ Pas de calcul de tendance (niveau gelé à la moyenne 90j)
- ✅ Seule la saisonnalité hebdomadaire varie (lundi ≠ dimanche)
- ✅ Impossible de dériver vers zéro ou l'infini
- ✅ Préserve les variations réalistes (±30 admissions/jour entre jours de semaine)

### 2. Changement du Modèle par Défaut

Modification de `predict_admissions_best()` :
```python
def predict_admissions_best(
    series, 
    horizon_jours=14,
    prefer="holt_winters_stable"  # ← Changé (était "best_by_backtest")
):
```

**Justification** : On privilégie la **cohérence physique** (hôpital ne peut pas tendre vers zéro patient) sur la **précision métrique** du backtest. Pour un système hospitalier en régime établi, la stabilité est plus importante que la minimisation de l'erreur à court terme.

## 📊 Résultats de Validation

### Test sur 180 jours (6 mois)

**Admissions** :
- Moyenne : **345.6/jour** (historique : 345.1) → Écart de 0.1%
- Moyennes mensuelles : 346.6 → 345.0 → 344.6 → 347.7 → 346.6 → 342.8
- Tendance : **-1.1%** sur 6 mois (négligeable)
- Écart max entre mois : 4.9 admissions/jour (1.4%)

**Occupation** :
- Occupation attendue en régime permanent : **1359 lits** (historique : 1357)
- Écart : **2 lits** (0.1%)
- DMS implicite : 3.93 jours (calculé automatiquement)

### Diagnostic : ✅ STABLE

```
✅ STABLE : Les admissions prédites restent stables autour de la moyenne
   Moyenne globale : 345.6/jour (historique : 345.1)
   La variation observée (4.9) est due à la saisonnalité hebdomadaire.

✅ Occupation attendue cohérente avec l'historique
```

## ⚠️ Piège Évité : Saisonnalité vs Tendance

**Erreur initiale** : Comparer `pred[jour_1]` vs `pred[jour_180]` (355.5 → 289.3 = -66 admissions)

**Réalité** : 
- Jour 1 = Lundi (saisonnalité : +19 admissions)
- Jour 180 = Dimanche (saisonnalité : -56 admissions)
- Différence de 75 admissions due au **jour de la semaine**, PAS à une tendance !

**Méthode correcte** : Comparer les **moyennes mensuelles** pour lisser la saisonnalité hebdomadaire.

## 📈 Impact Dashboard

L'utilisateur verra maintenant dans l'onglet **Prévisions** :
- Courbes d'admissions qui **oscillent** autour de 345/jour (saisonnalité hebdo)
- Pas de décroissance continue sur plusieurs mois
- Occupation stable autour de 1350 lits
- Intervalles de confiance réalistes (±60 lits, basés sur σ historique)

## 🔧 Fichiers Modifiés

1. **`src/prediction/models.py`** :
   - Ligne 112-160 : Réécriture complète de `predict_holt_winters()`
   - Ligne 563-590 : Changement du défaut de `predict_admissions_best()`

2. **`tests/debug_predictions.py`** :
   - Analyse par moyennes mensuelles (au lieu de jour 1 vs jour 180)
   - Diagnostic basé sur la tendance des moyennes (±2% = stable)

## 🎓 Leçons pour la Soutenance

1. **Métriques vs Réalisme** : Un modèle avec MAE=21.7 peut être **moins bon** qu'un modèle avec MAE=51.7 si le premier viole les contraintes physiques du système.

2. **Horizon de validation** : Le backtest sur 90 jours ne détecte pas les dérives qui apparaissent sur 6-12 mois.

3. **Saisonnalité hebdomadaire hospitalière** : 
   - Lundi/Mardi : +40-45 admissions (+12%)
   - Samedi/Dimanche : -36-56 admissions (-16%)
   - Total : ±20% de variation autour de la moyenne

4. **Contrainte d'équilibre** : Pour un hôpital en régime établi (pas de crise, pas de nouveau service), les admissions doivent rester autour de leur moyenne historique à ±5% près.

## ✅ Checklist Soutenance

- [x] Prédictions stables sur 180 jours (variation < 2%)
- [x] Saisonnalité hebdomadaire préservée
- [x] Occupation cohérente avec historique (écart < 1%)
- [x] Pas de dérive vers zéro
- [x] DMS calculée automatiquement (3.93 jours)
- [x] Dashboard fonctionnel
- [x] Documentation complète

---

**Date de correction** : 5 février 2026, 11h30  
**Modèle retenu** : Holt-Winters stable (niveau figé + saisonnalité hebdo)  
**Validation** : 180 jours, stabilité ±1.1%  
**Statut** : ✅ Prêt pour soutenance
