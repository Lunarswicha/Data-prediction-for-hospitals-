#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test rapide du système d'ensemble pour vérifier que tout fonctionne.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load import load_occupation
from src.prediction.ensemble import predict_admissions_ensemble, get_ensemble_info
import pandas as pd

def test_ensemble():
    print("\n" + "="*60)
    print("TEST DU SYSTÈME D'ENSEMBLE")
    print("="*60)
    
    # Charger les données
    print("\n1. Chargement des données...")
    occupation_df = load_occupation()
    print(f"   ✓ {len(occupation_df)} jours de données chargées")
    
    # Préparer la série
    print("\n2. Préparation de la série d'admissions...")
    adm_series = occupation_df.set_index("date")["admissions_jour"]
    if isinstance(adm_series, pd.DataFrame):
        adm_series = adm_series.squeeze()
    print(f"   ✓ Série de {len(adm_series)} observations")
    print(f"   Moyenne: {adm_series.mean():.1f} admissions/jour")
    print(f"   Écart-type: {adm_series.std():.1f}")
    
    # Test prédiction ensemble
    print("\n3. Prédiction par ensemble (30 jours)...")
    try:
        pred_adm = predict_admissions_ensemble(adm_series, horizon_jours=30)
        print(f"   ✓ Prédictions générées: {len(pred_adm)} jours")
        print(f"   Moyenne prévue: {pred_adm['prediction'].mean():.1f} admissions/jour")
        print(f"   IC bas moyen: {pred_adm['prediction_low'].mean():.1f}")
        print(f"   IC haut moyen: {pred_adm['prediction_high'].mean():.1f}")
    except Exception as e:
        print(f"   ✗ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Informations sur l'ensemble
    print("\n4. Informations sur l'ensemble...")
    try:
        info = get_ensemble_info(adm_series)
        print(f"   ✓ Validation: {info['validation_days']} jours")
        print(f"   Top 3 modèles retenus:")
        for nom in info['top3_names']:
            poids = info['weights'].get(nom, 0)
            perf = info['all_performances'].get(nom, {})
            print(f"      - {nom}: {poids:.0f}% (±10%: {perf.get('pct_within_10', 0):.1f}%, MAE: {perf.get('mae', 0):.1f})")
    except Exception as e:
        print(f"   ✗ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("✅ TOUS LES TESTS RÉUSSIS")
    print("="*60 + "\n")
    return True

if __name__ == "__main__":
    success = test_ensemble()
    sys.exit(0 if success else 1)
