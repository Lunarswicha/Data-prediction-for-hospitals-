# Diagnostic des prédictions — Patterns répétitifs

**Date** : 5 février 2026  
**Analyste** : Audit technique  
**Statut** : 🔴 PROBLÈME CRITIQUE IDENTIFIÉ

---

## 1. Symptômes observés

Les prédictions d'occupation des lits présentent des **patterns répétitifs incohérents** :
- Les valeurs prédites varient très peu d'un jour à l'autre
- Les prévisions semblent "plates" ou répètent un motif constant
- Pas de réponse réaliste aux variations d'admissions prédites

---

## 2. Cause racine identifiée

### Localisation du bug
**Fichier** : `src/prediction/models.py`  
**Fonction** : `predict_occupation_from_admissions()` (lignes ~750-800)

### Le problème

La formule utilisée pour calculer l'occupation prédite est **fondamentalement incorrecte** :

```python
# CODE ACTUEL (DÉFECTUEUX)
occ_pred = occ_mean * 0.85 + adm_val * ratio * 0.15 * duree
```

#### Pourquoi c'est incorrect ?

1. **Pas un modèle stock-flux** : Cette formule ne simule PAS l'évolution dynamique du stock de patients
2. **Poids arbitraires** : 85% de moyenne historique + 15% d'admissions = coefficients sortis de nulle part
3. **Pas de mémoire** : Chaque jour est calculé indépendamment, sans tenir compte du stock de la veille
4. **Patterns répétitifs** : Comme 85% du résultat est toujours `occ_mean` (constant), les variations sont minimes

#### Ce que devrait faire un vrai modèle stock

Un modèle stock-flux hospitalier correct doit simuler :

```
Stock(t) = Stock(t-1) + Entrées(t) - Sorties(t)

où :
  - Stock(t-1) = nombre de patients présents la veille
  - Entrées(t) = admissions du jour (prédites)
  - Sorties(t) = patients qui sortent (fonction de la durée de séjour)
```

Le taux de sortie peut être modélisé par :
```
Sorties(t) = Stock(t-1) / DMS(t)
```
où DMS = durée moyenne de séjour (avec variation saisonnière)

---

## 3. Impact

### Gravité : 🔴 CRITIQUE

- ❌ **Prévisions inexploitables** : Les patterns répétitifs ne reflètent pas la réalité
- ❌ **Modèle incohérent** : Ne respecte pas les principes de modélisation stock-flux
- ❌ **Perte de crédibilité** : Des décideurs verraient immédiatement que "ça ne marche pas"
- ⚠️ **Compromet tout le projet** : Pour une soutenance cet après-midi, c'est bloquant

### Ce qui fonctionne encore

- ✅ Les modèles de prédiction des **admissions** (Holt-Winters, Ridge, SARIMA) sont corrects
- ✅ La prédiction **directe de l'occupation** (`predict_occupation_direct`) est correcte
- ✅ La génération de données synthétiques est cohérente
- ✅ Le dashboard et les visualisations fonctionnent

---

## 4. Solution recommandée

### Correction du modèle stock

Remplacer la formule statique par une **simulation dynamique jour par jour** :

```python
def predict_occupation_from_admissions_CORRECT(...):
    """Modèle stock-flux avec simulation dynamique."""
    
    # Initialisation : stock actuel (dernier jour observé)
    stock_actuel = occ.iloc[-1]
    
    predictions = []
    for jour in range(horizon):
        # 1. Entrées du jour (admissions prédites)
        entrees = admissions_pred[jour]
        
        # 2. Durée de séjour saisonnière
        dms = _duree_sejour_saisonniere(mois_du_jour, base=6.0)
        
        # 3. Sorties = stock / DMS (modèle exponentiel)
        sorties = stock_actuel / dms
        
        # 4. Nouveau stock
        stock_actuel = max(0, stock_actuel + entrees - sorties)
        
        predictions.append(stock_actuel)
    
    return predictions
```

### Avantages de cette approche

- ✅ **Physiquement cohérent** : Respecte la conservation du stock
- ✅ **Dynamique** : Chaque jour dépend du précédent
- ✅ **Réaliste** : Répond aux variations d'admissions
- ✅ **Standard** : Utilisé dans la littérature (Lequertier 2022)

---

## 5. Actions immédiates

### Avant la soutenance

1. **Corriger `predict_occupation_from_admissions()`** avec le modèle stock dynamique
2. **Tester** sur quelques horizons (14j, 30j, 60j)
3. **Vérifier** que les courbes sont cohérentes (pas de patterns répétitifs)
4. **Régénérer** les graphiques du dashboard

### Temps estimé
⏱️ **15-20 minutes** pour la correction + tests

### Risques
- 🟢 **Faible** : La correction est localisée à une fonction
- 🟢 **Pas de casse** : Les autres modèles et le dashboard ne sont pas touchés
- 🟢 **Réversible** : On peut revenir en arrière si problème

---

## 6. Recommandations post-soutenance

1. **Validation croisée** : Comparer occupation prédite (via admissions) vs prédiction directe
2. **Backtest** : Évaluer le modèle stock sur données historiques
3. **Intervalles de confiance** : Propager l'incertitude des admissions au stock
4. **Documentation** : Ajouter formules et références dans le rapport de conception

---

## 7. Pourquoi Cursor a fait ça ?

Hypothèses sur l'origine du bug :

1. **Mauvaise compréhension du modèle stock** : Confusion entre régression statique et simulation dynamique
2. **Sur-optimisation prématurée** : Tentative de "lisser" les prévisions pour éviter des variations
3. **Copy-paste de code non adapté** : Formule heuristique d'un autre contexte
4. **Manque de validation** : Pas de test visuel des courbes générées

---

## Conclusion

🎯 **Diagnostic final** : Bug critique dans le modèle stock-flux  
🔧 **Correction** : Remplacer formule statique par simulation dynamique  
⏰ **Urgence** : À corriger avant la soutenance (15-20 min)  
✅ **Faisabilité** : Correction simple, risque faible, impact élevé

**Le reste du projet est solide** — c'est LA correction à faire pour sauver la démo.

---

*Diagnostic effectué : 5 février 2026*
