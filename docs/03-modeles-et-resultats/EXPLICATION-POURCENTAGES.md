# Explication des pourcentages affichés par les modèles

Ce document décrit **quels pourcentages** sortent des différents modèles, **comment ils sont calculés**, et **pourquoi ils sont souvent élevés** (surtout sur données synthétiques).

---

## 1. Les deux familles de % affichés

| Type | Où on le voit | Signification |
|------|----------------|----------------|
| **Taux d'occupation** (ex. 67 %, 82 %) | Prévisions, graphiques occupation, métrique "Taux d'occupation max prévu" | Part des lits occupés : `occupation_lits / 1800` |
| **% à ±10 %** (ex. 85 %, 92 %) | Validation du modèle, Backtest, onglet Boosting | Proportion de jours où l'erreur relative de prévision est ≤ 10 % (réf. Bouteloup) |

Les **seuils 85 % et 95 %** (alerte / critique) sont des **constantes de gestion**, pas une sortie de modèle : ils servent à déclencher des alertes quand le **taux d'occupation** les dépasse.

---

## 2. Taux d'occupation (pourquoi souvent entre 65 % et 85 %)

### Calcul

- **Formule** : `taux_occupation = occupation_lits / capacite_lits` avec `capacite_lits = 1800`.
- Les modèles prédisent soit directement **occupation_lits** (prédiction directe), soit **admissions** puis on en déduit l'occupation via un modèle de stock (durée de séjour, etc.). Dans les deux cas on affiche le **taux** = prévision en lits / 1800.

### Pourquoi c'est "élevé" en moyenne

1. **Données synthétiques calibrées "réalistes"**  
   Le générateur (`src.data.generate`) produit des admissions quotidiennes (~320/jour en base) et une occupation dérivée d'un stock avec durée de séjour moyenne. Le niveau est volontairement proche de ce qu'on observe dans un grand hôpital : **taux d'occupation souvent entre 60 % et 80 %**.

2. **Les modèles reproduisent la dynamique**  
   Holt-Winters, Ridge, SARIMA, etc. s'ajustent à cette série. Ils ne "gonflent" pas artificiellement le taux : ils **prévoient un niveau cohérent avec l'historique**. Si l'historique tourne autour de 70 %, les prévisions tournent aussi autour de 70 %.

3. **Seuils 85 % et 95 %**  
   Ce sont des **seuils d'alerte** (config) : au-dessus de 85 % = alerte, au-dessus de 95 % = critique. Quand le taux prévu est "élevé", c'est donc soit que les données synthétiques sont chargées, soit que la prévision anticipe une période tendue ; les seuils servent justement à signaler ces cas.

En résumé : **un taux d'occupation souvent "élevé" reflète le calibrage des données et la cohérence des modèles avec l'historique**, pas un biais systématique des algorithmes.

---

## 3. % à ±10 % (Bouteloup) — pourquoi souvent au-dessus de 80 %

### Calcul

- On fixe une **période de validation** (ex. les 90 derniers jours).
- Pour chaque jour de cette période, on compare la **prévision** (admissions ou occupation selon le contexte) à la **valeur observée**.
- **Erreur relative** : `|prévu - observé| / observé` (en évitant division par zéro).
- Un jour compte comme "réussi" si cette erreur **≤ 10 %**.
- Le **% à ±10 %** = (nombre de jours "réussis") / (nombre total de jours de validation), donc une proportion entre 0 % et 100 %.

Exemple : 90 jours de test, 78 jours avec erreur ≤ 10 % → **% à ±10 % = 78/90 ≈ 86,7 %**.

### Pourquoi ce % est souvent élevé (85–95 %) sur ce projet

1. **Données synthétiques lisses**  
   Le jeu est généré avec tendance, saisonnalité hebdo/mensuelle et peu de chocs. Les séries sont **faciles à prévoir** pour des modèles comme Holt-Winters ou Ridge : beaucoup de jours tombent naturellement dans la bande ±10 %.

2. **Référence littérature (Bouteloup)**  
   Sur **données réelles** (urgences, flux hospitaliers), atteindre 70–80 % de jours à ±10 % est déjà considéré comme bon. Ici, **80–95 % est attendu** car la difficulté de prévision est moindre.

3. **Intérêt de la métrique**  
   Elle sert à **comparer les modèles** (backtest, onglet Boosting) et à voir si, en passant à de vraies données, le % baisse. Une baisse à 70–75 % sur du réel resterait cohérente avec la littérature.

En résumé : **un % à ±10 % élevé sur données synthétiques est normal ; sur données réelles, on s'attend à des valeurs plus basses (ordre 70–85 % selon la complexité du flux).**

---

## 4. Autres % affichés (sous-estimation / surestimation)

Dans la validation et le backtest on affiche aussi :

- **Sous-estimation** : % de jours où `prévu < 0,9 × observé`.
- **Surestimation** : % de jours où `prévu > 1,1 × observé`.
- **Biais moyen** : moyenne de (prévu − observé) ; positif = en moyenne le modèle surestime.

Ils permettent de voir si le modèle a tendance à être trop prudent (sous-estimation) ou trop optimiste (surestimation), et complètent le % à ±10 %.

---

## 5. Synthèse

| % affiché | Signification | Pourquoi souvent "élevé" |
|-----------|----------------|--------------------------|
| **Taux d'occupation** (ex. 70 %) | Lits occupés / 1800 | Données synthétiques calibrées pour un hôpital chargé ; les modèles suivent l'historique. |
| **% à ±10 %** (ex. 88 %) | Part des jours avec erreur relative ≤ 10 % | Série synthétique lisse et prévisible ; sur données réelles, 70–80 % serait déjà bon. |
| **85 % / 95 %** (seuils) | Alerte / critique | Constantes de gestion, pas une sortie de modèle. |

Les modèles ne "fabriquent" pas des % anormalement hauts : ils reflètent le **niveau et la prévisibilité** du jeu de données (ici synthétique). Avec des données réelles, les taux d'occupation dépendront du terrain, et le % à ±10 % aura tendance à diminuer.
