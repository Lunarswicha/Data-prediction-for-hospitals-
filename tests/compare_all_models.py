"""Test : Vérifier que chaque modèle produit des prédictions DIFFÉRENTES."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from src.data.load import load_occupation
from src.prediction.models import (
    predict_holt_winters,
    predict_regression,
    predict_sarima,
    predict_boosting,
    predict_moving_average,
    prepare_series,
)

print("="*80)
print("TEST : COMPARAISON DE TOUS LES MODÈLES")
print("="*80)

# Charger les données
occupation_df = load_occupation(ROOT / 'data')
adm = prepare_series(occupation_df, "admissions_jour")

horizon = 30  # 30 jours pour comparaison rapide

print(f"\nDonnées : {len(adm)} jours, horizon = {horizon} jours")
print(f"Moyenne historique (30 derniers jours) : {adm.iloc[-30:].mean():.1f}")

# Tester TOUS les modèles
modeles = {
    "Holt-Winters": predict_holt_winters,
    "Régression": predict_regression,
    "SARIMA": predict_sarima,
    "Boosting": predict_boosting,
    "Moving Average": predict_moving_average,
}

resultats = {}
for nom, fonction in modeles.items():
    print(f"\n{'='*80}")
    print(f"MODÈLE : {nom}")
    print(f"{'='*80}")
    try:
        pred = fonction(adm, horizon_jours=horizon)
        if pred is not None and len(pred) > 0:
            values = pred["prediction"].values if "prediction" in pred.columns else pred.values
            resultats[nom] = values
            
            print(f"✅ Prédictions générées : {len(values)} jours")
            print(f"   Jour 1 : {values[0]:.1f}")
            print(f"   Jour 15 : {values[14]:.1f}")
            print(f"   Jour 30 : {values[29]:.1f}")
            print(f"   Moyenne : {values.mean():.1f}")
            print(f"   Écart-type : {values.std():.1f}")
            print(f"   Min : {values.min():.1f}, Max : {values.max():.1f}")
        else:
            print(f"❌ Échec : retour None ou vide")
            resultats[nom] = None
    except Exception as e:
        print(f"❌ ERREUR : {e}")
        resultats[nom] = None

# COMPARAISON : Les modèles sont-ils DIFFÉRENTS ?
print("\n" + "="*80)
print("ANALYSE : LES MODÈLES SONT-ILS DIFFÉRENTS ?")
print("="*80)

modeles_valides = {k: v for k, v in resultats.items() if v is not None}

if len(modeles_valides) < 2:
    print("❌ Moins de 2 modèles valides, impossible de comparer")
else:
    print(f"\n{len(modeles_valides)} modèles valides à comparer\n")
    
    # Comparer chaque paire de modèles
    noms = list(modeles_valides.keys())
    for i in range(len(noms)):
        for j in range(i+1, len(noms)):
            nom1, nom2 = noms[i], noms[j]
            pred1, pred2 = modeles_valides[nom1], modeles_valides[nom2]
            
            # Calculer la différence
            diff_abs_mean = np.abs(pred1 - pred2).mean()
            diff_pct = (diff_abs_mean / pred1.mean()) * 100
            
            # Corrélation
            correlation = np.corrcoef(pred1, pred2)[0, 1]
            
            # Identiques ?
            identiques = np.allclose(pred1, pred2, atol=0.1)
            
            print(f"{nom1} vs {nom2}:")
            print(f"   Différence moyenne : {diff_abs_mean:.2f} ({diff_pct:.1f}%)")
            print(f"   Corrélation : {correlation:.3f}")
            
            if identiques:
                print(f"   ⚠️  IDENTIQUES (à 0.1 près) !")
            elif diff_pct < 1:
                print(f"   ⚠️  QUASI-IDENTIQUES (< 1% de différence)")
            elif correlation > 0.95:
                print(f"   ⚠️  TRÈS CORRÉLÉS (forme similaire)")
            else:
                print(f"   ✅ DIFFÉRENTS")
            print()

# VERDICT FINAL
print("="*80)
print("VERDICT")
print("="*80)

# Calculer la variance des moyennes de tous les modèles
moyennes = [v.mean() for v in modeles_valides.values()]
ecart_moyennes = max(moyennes) - min(moyennes)
variance_relative = ecart_moyennes / np.mean(moyennes) * 100

print(f"\nMoyennes prédites : {[f'{m:.1f}' for m in moyennes]}")
print(f"Écart max entre modèles : {ecart_moyennes:.1f} ({variance_relative:.1f}%)")

if variance_relative < 2:
    print("\n❌ PROBLÈME CONFIRMÉ : Tous les modèles donnent des résultats quasi-identiques")
    print("   Cela suggère un problème de hard-coding ou de duplication de code")
elif variance_relative < 10:
    print("\n⚠️  Modèles peu différenciés (< 10% d'écart)")
    print("   Les modèles capturent peut-être la même structure de données")
else:
    print("\n✅ Modèles suffisamment différenciés")
    print("   Chaque modèle apporte une perspective différente")
