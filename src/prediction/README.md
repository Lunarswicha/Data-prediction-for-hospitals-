# Module prediction

- **Modèles** : Holt-Winters (saisonnalité 7j), régression Ridge (lags 1–7–14 + **mean j-7 à j-13**, **jours fériés, vacances scolaires, température synth.** — réf. Bouteloup), SARIMA en option, moyenne glissante en fallback.
- **Durée de séjour saisonnière** (réf. Lequertier) dans le modèle stock (hiver +8 %, été −8 %).
- Prévision des **admissions** et de l'occupation des lits (directe ou via modèle stock), avec **intervalles de confiance** (IC 95 %) lorsque possible.
- **Validation** : métrique « % de jours à ±10 % » et biais (réf. Bouteloup), exposée dans le dashboard.

Consignes : « Développement d'un/de modèle(s) de prédiction permettant d'anticiper les pics d'activité ».
