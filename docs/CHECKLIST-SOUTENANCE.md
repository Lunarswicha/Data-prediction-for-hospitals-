# ⚡ CHECKLIST RAPIDE AVANT SOUTENANCE

**🕐 Temps estimé : 5-10 minutes**

---

## 1️⃣ Vérification visuelle du dashboard (5 min)

```bash
cd "/Users/lunarswicha/Desktop/Data Hopital"
streamlit run app/dashboard.py
```

### À vérifier dans chaque section :

#### 📊 Flux & historique
- [ ] Les graphiques s'affichent correctement
- [ ] Les données vont bien de 2022 à 2024
- [ ] Les filtres fonctionnent

#### 🔮 Prévisions
- [ ] **CRITIQUE** : Les courbes d'occupation sont **dynamiques** (pas plates)
- [ ] Horizon 14j, 30j, 60j fonctionnent tous
- [ ] Les intervalles de confiance s'affichent (zone grisée)
- [ ] Le tableau de détail affiche les alertes (normal/alerte/critique)
- [ ] Le backtest affiche une comparaison prévu vs observé

#### 🎲 Simulation de scénarios
- [ ] Les 4 scénarios se lancent (épidémie, grève, canicule, afflux)
- [ ] Les graphiques montrent bien une évolution différente selon le scénario
- [ ] L'export CSV fonctionne

#### 🤖 Modèle Boosting
- [ ] Les métriques s'affichent (MAE, RMSE, % ±10%)
- [ ] La comparaison avec le modèle principal est visible

#### 💡 Recommandations
- [ ] Des recommandations s'affichent (renforts, vigilance, report activité, etc.)
- [ ] La priorisation est visible (Critique / Urgent / Normal)

---

## 2️⃣ Points à retenir pour la soutenance (1 min)

### Modèle stock-flux (si question sur les prévisions)

> "Nous utilisons un modèle stock-flux dynamique pour prédire l'occupation jour par jour :  
> **Stock(t) = Stock(t-1) + Entrées(t) - Sorties(t)**  
> où les sorties dépendent de la durée moyenne de séjour, calculée automatiquement depuis les données historiques."

### Validation (si question sur la qualité)

> "Sur un horizon de 14 jours, **92% des jours** présentent des variations significatives, ce qui démontre la capacité du modèle à capturer la dynamique réelle."

### DMS (si question technique)

> "La durée moyenne de séjour est **calculée automatiquement** : Stock moyen / Admissions moyennes ≈ 4 jours, avec une variation saisonnière (plus longue en hiver, plus courte en été)."

---

## 3️⃣ Slides recommandées (optionnel, 2 min)

### Slide "Modèles de prévision"

Deux approches :
1. **Prédiction directe** de l'occupation (Holt-Winters, Ridge, SARIMA)
2. **Modèle stock-flux** : prédiction des admissions → simulation de l'occupation

Formule :
```
Stock(t) = min(Stock(t-1) + Admissions(t) - Sorties(t), Capacité)
Sorties(t) = Stock(t-1) / DMS(t)
```

### Slide "Validation"

| Métrique | Résultat |
|----------|----------|
| % jours avec variations >0.5% | **92%** |
| Occupation prédite (14j) | 1454-1577 lits (81-88%) |
| Taux d'erreur ±10% | ~85-90% (réf. Bouteloup 84%) |

### Slide "Dashboard"

Screenshot de l'onglet "Prévisions" avec :
- Courbe d'occupation prédite
- Intervalles de confiance
- Seuils d'alerte (85% / 95%)
- Recommandations

---

## 4️⃣ Réponses aux questions potentielles

### "Comment gérez-vous les pics épidémiques ?"

> "Nos modèles intègrent la **saisonnalité** (mois, jour de semaine), les **jours fériés**, et la **température synthétique**. De plus, l'onglet **Simulation** permet de tester 4 scénarios : épidémie grippe, grève, canicule, et afflux massif, avec des paramètres configurables (durée, intensité)."

### "Quelle est la source de vos données ?"

> "Les données sont **100% synthétiques**, générées pour reproduire les tendances réalistes d'un grand hôpital (Pitié-Salpêtrière) : saisonnalité hiver/été, répartition par service, jour de la semaine. Aucune donnée réelle de patients n'est utilisée (conformité RGPD)."

### "Avez-vous comparé vos modèles ?"

> "Oui, nous effectuons un **backtest** : le modèle est entraîné sur le passé et testé sur les 90 derniers jours. Nous comparons 4 familles de modèles (Holt-Winters, Ridge, SARIMA, Boosting) selon la métrique **% à ±10%** (référence littérature Bouteloup 2020) et la MAE."

### "Quelle est la précision de vos prévisions ?"

> "Sur données synthétiques (lisses), nous atteignons **85-95% de jours à ±10%**. Sur données réelles, la littérature (Bouteloup 2020, urgences Pellegrin) rapporte **83-84%**. Nos modèles sont donc dans l'ordre de grandeur attendu."

### "Comment passeriez-vous en production ?"

> "Il faudrait :  
> 1. **Données réelles** : PMSI, RPU (avec autorisation CEREES, CNIL)  
> 2. **Hébergement sécurisé** : HDS (Hébergeur de Données de Santé)  
> 3. **Ré-entraînement** des modèles sur les données historiques réelles  
> 4. **Monitoring** : alertes en cas de dérive des prévisions  
> 5. **Formation** des utilisateurs (direction, cadres de santé)"

---

## 5️⃣ En cas de problème technique en direct

### Le dashboard ne se lance pas
```bash
# Vérifier que vous êtes dans le bon répertoire
cd "/Users/lunarswicha/Desktop/Data Hopital"

# Vérifier les dépendances
pip install -r requirements.txt

# Relancer
streamlit run app/dashboard.py
```

### Une erreur s'affiche dans l'onglet Prévisions
- Réduire l'horizon (essayer 14j au lieu de 60j)
- Changer de modèle (essayer "Modèle automatique" ou "Holt-Winters")
- Passer par l'onglet "Modèle Boosting" qui est indépendant

### Les graphiques ne s'affichent pas
- Rafraîchir la page (F5 ou Cmd+R)
- Vérifier que vous avez bien généré les données : `python -m src.data.generate`

---

## ✅ C'est prêt !

**Vous avez :**
- [x] Un projet complet et fonctionnel
- [x] Une documentation exhaustive
- [x] Des modèles validés (92% variations dynamiques)
- [x] Un dashboard professionnel
- [x] Une référence solide à la littérature

**Ce qui peut faire la différence :**
- 🎯 **Honnêteté** : dire que les données sont synthétiques, que la validation opérationnelle nécessiterait des données réelles
- 🎯 **Rigueur** : expliquer les choix méthodologiques (pourquoi Holt-Winters, pourquoi l'IC 95%, pourquoi les seuils 85%/95%)
- 🎯 **Perspective** : montrer que vous avez conscience des limites et des pistes d'évolution

---

**🚀 BONNE SOUTENANCE !**

*"Le problème a été identifié et corrigé. Le projet est maintenant solide, cohérent, et prêt à être présenté."*

---

📁 **Fichiers de référence** :
- Résumé complet : [`docs/RESUME-FINAL.md`](RESUME-FINAL.md)
- Diagnostic bug : [`docs/DIAGNOSTIC-PREDICTIONS.md`](DIAGNOSTIC-PREDICTIONS.md)
- Correction validée : [`docs/CORRECTION-VALIDEE.md`](CORRECTION-VALIDEE.md)
