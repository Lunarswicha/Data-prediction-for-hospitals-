"""Diagnostic complet : pourquoi l'occupation monte à 100% ?"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.load import load_occupation
from src.prediction.models import predict_admissions_best, predict_occupation_from_admissions, prepare_series

occupation_df = load_occupation(ROOT / 'data')
adm = prepare_series(occupation_df, "admissions_jour")
occ = prepare_series(occupation_df, "occupation_lits")

print("="*80)
print("DIAGNOSTIC : POURQUOI L'OCCUPATION MONTE À 100% ?")
print("="*80)

# 1. Données historiques
print("\n[1] DONNÉES HISTORIQUES (90 derniers jours)")
print(f"Admissions moyennes : {adm.iloc[-90:].mean():.1f}/jour")
print(f"Occupation moyenne : {occ.iloc[-90:].mean():.0f} lits ({occ.iloc[-90:].mean()/1800*100:.0f}%)")
dms_implicite = occ.iloc[-90:].mean() / adm.iloc[-90:].mean()
print(f"DMS implicite : {dms_implicite:.2f} jours")

# 2. Admissions prédites
print("\n[2] ADMISSIONS PRÉDITES (180 jours)")
pred_adm = predict_admissions_best(adm, horizon_jours=180)
adm_pred = pred_adm["prediction"].values
print(f"Moyenne : {adm_pred.mean():.1f}/jour (historique : {adm.iloc[-90:].mean():.1f})")
print(f"Min : {adm_pred.min():.1f}, Max : {adm_pred.max():.1f}")

# 3. Occupation attendue en régime permanent
print("\n[3] OCCUPATION ATTENDUE (Stock = Admissions × DMS)")
occ_attendue = adm_pred.mean() * dms_implicite
print(f"Si admissions = {adm_pred.mean():.1f}/jour et DMS = {dms_implicite:.2f}j")
print(f"=> Occupation = {occ_attendue:.0f} lits ({occ_attendue/1800*100:.0f}%)")

if occ_attendue > 1800:
    print(f"❌ PROBLÈME : Occupation attendue DÉPASSE la capacité de {occ_attendue - 1800:.0f} lits !")
elif occ_attendue > 1700:
    print(f"⚠️  Occupation attendue très élevée (> 94%) → proche de la saturation")

# 4. Prédiction occupation réelle
print("\n[4] PRÉDICTION OCCUPATION (via modèle stock-flux)")
pred_occ = predict_occupation_from_admissions(occupation_df, horizon_jours=180)
occ_pred = pred_occ["occupation_lits_pred"].values
occ_high = pred_occ["occupation_lits_high"].values

print(f"Moyenne : {occ_pred.mean():.0f} lits ({occ_pred.mean()/1800*100:.0f}%)")
print(f"Max : {occ_pred.max():.0f} lits ({occ_pred.max()/1800*100:.0f}%)")
print(f"IC haut max : {occ_high.max():.0f} lits ({occ_high.max()/1800*100:.0f}%)")

# 5. Diagnostic
print("\n" + "="*80)
print("DIAGNOSTIC")
print("="*80)

if occ_pred.max() >= 1800:
    print("❌ SATURATION : Prédiction atteint 100% de la capacité")
    print("   Causes possibles :")
    if adm_pred.mean() > adm.iloc[-90:].mean() * 1.1:
        print(f"   - Admissions prédites trop élevées ({adm_pred.mean():.1f} vs {adm.iloc[-90:].mean():.1f} historique)")
    if dms_implicite > 5:
        print(f"   - DMS trop longue ({dms_implicite:.2f} jours)")
    print(f"   - Le modèle stock accumule sans sorties suffisantes")
elif occ_pred.max() > 1700:
    print(f"⚠️  Occupation élevée mais gérable (max {occ_pred.max()/1800*100:.0f}%)")
else:
    print(f"✅ Occupation réaliste (max {occ_pred.max()/1800*100:.0f}%)")

# Convergence vers le régime permanent ?
debut = occ_pred[:30].mean()
fin = occ_pred[-30:].mean()
print(f"\nConvergence : début (30j) = {debut:.0f} lits, fin (30j) = {fin:.0f} lits")
if abs(fin - debut) < 50:
    print("✅ Le modèle converge vers un équilibre stable")
elif fin > debut:
    print(f"⚠️  Le modèle diverge vers le haut (+{fin-debut:.0f} lits sur 180j)")
else:
    print(f"⚠️  Le modèle diverge vers le bas ({fin-debut:.0f} lits sur 180j)")
