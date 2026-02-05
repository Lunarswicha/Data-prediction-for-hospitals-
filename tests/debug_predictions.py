"""Debug : Analyser pourquoi les prédictions décroissent continuellement."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from src.data.load import load_occupation
from src.prediction.models import (
    predict_admissions_best,
    predict_occupation_from_admissions,
    prepare_series,
)

print("="*70)
print("DEBUG : ANALYSE DES PRÉDICTIONS")
print("="*70)

# Charger les données
occupation_df = load_occupation(ROOT / 'data')
adm = prepare_series(occupation_df, "admissions_jour")

# 1. Vérifier les admissions historiques récentes
print("\n[1] ADMISSIONS HISTORIQUES (30 derniers jours)")
adm_recent = adm.iloc[-30:]
print(f"Moyenne : {adm_recent.mean():.1f} admissions/jour")
print(f"Min : {adm_recent.min():.1f}")
print(f"Max : {adm_recent.max():.1f}")
print(f"Tendance : {adm_recent.iloc[-1] - adm_recent.iloc[0]:.1f} (dernier - premier)")

# 2. Prédire les admissions sur 180 jours AVEC HW STABLE (nouveau défaut)
print("\n[2] PRÉDICTION DES ADMISSIONS (180 jours)")
print("   Méthode : prefer='holt_winters_stable' (DÉFAUT)")
pred_adm = predict_admissions_best(adm, horizon_jours=180)  # Utilise HW stable par défaut

# Pour comparaison : afficher ce que le backtest aurait sélectionné
from src.prediction.models import select_best_model_by_backtest
best_name, metrics = select_best_model_by_backtest(adm, validation_days=90)
print(f"\nSi on utilisait best_by_backtest, le modèle sélectionné serait : {best_name.upper()}")
if metrics:
    for model, m in metrics.items():
        print(f"  {model}: MAE={m['mae']:.1f}, % ±10%={m['pct_within_10']*100:.1f}%")

adm_pred = pred_adm["prediction"].values

print(f"\nRésultats prédictions (HW stable) :")
print(f"Jour 1 : {adm_pred[0]:.1f}")
print(f"Jour 30 : {adm_pred[29]:.1f}")
print(f"Jour 60 : {adm_pred[59]:.1f}")
print(f"Jour 90 : {adm_pred[89]:.1f}")
print(f"Jour 180 : {adm_pred[179]:.1f}")

# IMPORTANT : Calculer la MOYENNE des prédictions, pas la différence premier-dernier
print(f"\n⚠️  ATTENTION : La différence jour 1 vs jour 180 reflète la saisonnalité hebdo,")
print(f"   pas une tendance. Le jour 180 pourrait tomber sur un dimanche (creux de semaine).")
print(f"\nMoyenne des prédictions sur 180 jours : {adm_pred.mean():.1f}")
print(f"Écart-type : {adm_pred.std():.1f}")
print(f"Min : {adm_pred.min():.1f}, Max : {adm_pred.max():.1f}")

# Tendance RÉELLE : comparer moyennes par mois
print(f"\nAnalyse par périodes de 30 jours (moyennes pour lisser la saisonnalité) :")
for i in range(0, 180, 30):
    segment = adm_pred[i:min(i+30, 180)]
    print(f"  Jours {i+1:3d}-{i+30:3d} : moyenne = {segment.mean():.1f}")

# 3. Prédire l'occupation via le modèle stock
print("\n[3] PRÉDICTION DE L'OCCUPATION (180 jours)")
pred_occ = predict_occupation_from_admissions(occupation_df, horizon_jours=180)
occ_pred = pred_occ["occupation_lits_pred"].values

print(f"Jour 1 : {occ_pred[0]:.0f} lits")
print(f"Jour 30 : {occ_pred[29]:.0f} lits")
print(f"Jour 60 : {occ_pred[59]:.0f} lits")
print(f"Jour 90 : {occ_pred[89]:.0f} lits")
print(f"Jour 180 : {occ_pred[179]:.0f} lits")
print(f"\nVariation totale : {occ_pred[-1] - occ_pred[0]:.0f} lits ({(occ_pred[-1] - occ_pred[0])/occ_pred[0]*100:.1f}%)")

# 4. Vérifier l'équilibre entrées/sorties
print("\n[4] ÉQUILIBRE ENTRÉES/SORTIES")
occ = prepare_series(occupation_df, "occupation_lits")
occ_hist_mean = occ.iloc[-90:].mean()
adm_hist_mean = adm.iloc[-90:].mean()
dms_implicite = occ_hist_mean / adm_hist_mean

print(f"Occupation historique moyenne (90j) : {occ_hist_mean:.0f} lits")
print(f"Admissions historiques moyennes (90j) : {adm_hist_mean:.1f}/jour")
print(f"DMS implicite : {dms_implicite:.2f} jours")
print(f"\n=> Régime permanent théorique : {adm_hist_mean:.1f} × {dms_implicite:.2f} = {adm_hist_mean * dms_implicite:.0f} lits")

# Si admissions prédites < admissions historiques => baisse unavoidable
adm_pred_mean = adm_pred.mean()
occ_attendue = adm_pred_mean * dms_implicite
print(f"\nAdmissions prédites moyennes : {adm_pred_mean:.1f}/jour")
print(f"=> Occupation attendue en régime permanent : {occ_attendue:.0f} lits")
print(f"=> Écart avec l'historique : {occ_attendue - occ_hist_mean:.0f} lits ({(occ_attendue - occ_hist_mean)/occ_hist_mean*100:.1f}%)")

# 5. Diagnostic
print("\n" + "="*70)
print("DIAGNOSTIC")
print("="*70)

# Comparer les moyennes des 6 périodes de 30 jours pour détecter une vraie tendance
moyennes_30j = [adm_pred[i:i+30].mean() for i in range(0, 180, 30)]
tendance_moyenne = moyennes_30j[-1] - moyennes_30j[0]
ecart_max = max(moyennes_30j) - min(moyennes_30j)

print(f"Moyennes mensuelles : {[f'{m:.1f}' for m in moyennes_30j]}")
print(f"Tendance (dernier mois - premier mois) : {tendance_moyenne:+.1f} ({tendance_moyenne/moyennes_30j[0]*100:+.1f}%)")
print(f"Écart max entre mois : {ecart_max:.1f} ({ecart_max/adm_pred_mean*100:.1f}%)")

if abs(tendance_moyenne) / adm_pred_mean < 0.02:  # < 2% de variation entre premier et dernier mois
    print("\n✅ STABLE : Les admissions prédites restent stables autour de la moyenne")
    print(f"   Moyenne globale : {adm_pred_mean:.1f}/jour (historique : {adm_hist_mean:.1f})")
    print(f"   La variation observée ({ecart_max:.1f}) est due à la saisonnalité hebdomadaire.")
elif tendance_moyenne < -adm_pred_mean * 0.05:  # Baisse > 5%
    print("\n❌ PROBLÈME : Les admissions prédites DÉCROISSENT de manière continue")
    print(f"   Baisse de {-tendance_moyenne:.1f} admissions/jour sur 6 mois ({tendance_moyenne/moyennes_30j[0]*100:+.1f}%)")
elif tendance_moyenne > adm_pred_mean * 0.05:  # Hausse > 5%
    print("\n⚠️  Les admissions prédites AUGMENTENT de manière continue")
    print(f"   Hausse de {tendance_moyenne:.1f} admissions/jour sur 6 mois ({tendance_moyenne/moyennes_30j[0]*100:+.1f}%)")
else:
    print("\n✅ Les admissions prédites sont raisonnablement stables")
    
# Vérifier la cohérence occupation
if abs(occ_attendue - occ_hist_mean) / occ_hist_mean > 0.10:
    print(f"\n⚠️  Écart significatif avec l'historique : {occ_attendue - occ_hist_mean:+.0f} lits ({(occ_attendue - occ_hist_mean)/occ_hist_mean*100:+.1f}%)")
else:
    print(f"\n✅ Occupation attendue cohérente avec l'historique")

