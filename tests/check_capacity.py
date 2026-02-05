"""Vérifier si les prédictions dépassent la capacité physique."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.load import load_occupation
from src.prediction.models import predict_occupation_from_admissions

occupation_df = load_occupation(ROOT / 'data')
pred = predict_occupation_from_admissions(occupation_df, horizon_jours=180)

print("="*80)
print("VÉRIFICATION CAPACITÉ PHYSIQUE")
print("="*80)

print(f"\nPrédictions occupation (180 jours) :")
print(f"Min : {pred['occupation_lits_pred'].min():.0f} lits")
print(f"Max : {pred['occupation_lits_pred'].max():.0f} lits")
print(f"Moyenne : {pred['occupation_lits_pred'].mean():.0f} lits")

print(f"\nBorne haute IC95% :")
print(f"Min : {pred['occupation_lits_high'].min():.0f} lits")
print(f"Max : {pred['occupation_lits_high'].max():.0f} lits")

print(f"\nBorne basse IC95% :")
print(f"Min : {pred['occupation_lits_low'].min():.0f} lits")
print(f"Max : {pred['occupation_lits_low'].max():.0f} lits")

print(f"\nCAPACITÉ MAXIMALE : 1800 lits")

print("\n" + "="*80)
print("DIAGNOSTIC")
print("="*80)

issues = []

if pred['occupation_lits_pred'].max() > 1800:
    over = pred['occupation_lits_pred'].max() - 1800
    issues.append(f"❌ Prédiction dépasse capacité de {over:.0f} lits ({pred['occupation_lits_pred'].max():.0f} lits)")

if pred['occupation_lits_high'].max() > 1800:
    over = pred['occupation_lits_high'].max() - 1800
    issues.append(f"❌ IC haut dépasse capacité de {over:.0f} lits ({pred['occupation_lits_high'].max():.0f} lits)")
    
if pred['occupation_lits_pred'].max() > 1800 * 0.98:
    issues.append(f"⚠️  Prédiction atteint {pred['occupation_lits_pred'].max()/1800*100:.0f}% de la capacité (saturation)")

if issues:
    for issue in issues:
        print(issue)
else:
    print("✅ Toutes les prédictions respectent la capacité physique")
