"""
Script de test rapide pour valider la correction du modèle stock-flux.
Exécuter : python tests/test_correction_stock.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data.load import load_admissions, load_occupation
from src.prediction.models import (
    predict_occupation_from_admissions,
    predict_occupation_direct,
    predict_admissions_best,
    prepare_series,
)

print("=" * 70)
print("TEST DE VALIDATION — CORRECTION DU MODÈLE STOCK-FLUX")
print("=" * 70)

# 1. Charger les données
print("\n[1/4] Chargement des données...")
admissions_df = load_admissions(ROOT / "data")
occupation_df = load_occupation(ROOT / "data")
print(f"✅ Données chargées : {len(occupation_df)} jours, du {occupation_df['date'].min()} au {occupation_df['date'].max()}")

# 2. Tester prédiction sur différents horizons
horizons = [14, 30, 60]
print(f"\n[2/4] Test sur {len(horizons)} horizons : {horizons} jours")

for h in horizons:
    print(f"\n--- Horizon {h} jours ---")
    pred_df = predict_occupation_from_admissions(
        occupation_df,
        horizon_jours=h,
        use_best_admissions=True,
        duree_sejour_saisonniere=True,
    )
    
    if pred_df.empty:
        print(f"❌ Échec pour horizon {h}j")
        continue
    
    # Vérifications
    print(f"✅ Prédiction générée : {len(pred_df)} jours")
    
    # 1. Pas de valeurs constantes répétées (symptôme de l'ancien bug)
    occ_vals = pred_df["occupation_lits_pred"].values
    variations = np.diff(occ_vals)
    pct_variations = np.abs(variations / occ_vals[:-1]) * 100
    
    # Si >80% des jours ont une variation <0.5%, c'est suspect
    faible_variation = (pct_variations < 0.5).sum() / len(pct_variations) * 100
    
    if faible_variation > 80:
        print(f"⚠️  ALERTE : {faible_variation:.1f}% des jours ont une variation <0.5% (patterns répétitifs possibles)")
    else:
        print(f"✅ Variations dynamiques : {100-faible_variation:.1f}% des jours varient de >0.5%")
    
    # 2. Occupation reste dans un intervalle plausible
    min_occ = occ_vals.min()
    max_occ = occ_vals.max()
    mean_occ = occ_vals.mean()
    
    print(f"   Occupation : min={min_occ:.0f}, max={max_occ:.0f}, moy={mean_occ:.0f} lits")
    
    if min_occ < 0:
        print(f"❌ ERREUR : valeurs négatives détectées !")
    elif min_occ == max_occ:
        print(f"❌ ERREUR : occupation constante sur {h} jours (bug pas corrigé !)")
    else:
        print(f"✅ Plage cohérente")
    
    # 3. Admissions prédites présentes
    if "admissions_pred" in pred_df.columns:
        adm_vals = pred_df["admissions_pred"].values
        print(f"   Admissions prédites : min={adm_vals.min():.0f}, max={adm_vals.max():.0f}")

# 3. Test visuel : comparaison ancien historique + prévisions
print(f"\n[3/4] Génération de graphiques de validation...")

horizon_test = 60
pred_stock = predict_occupation_from_admissions(occupation_df, horizon_jours=horizon_test, use_best_admissions=True)

if not pred_stock.empty:
    # Historique (30 derniers jours)
    hist = occupation_df.tail(30).copy()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Graphique 1 : Occupation (historique + prévision)
    ax1 = axes[0]
    ax1.plot(hist["date"], hist["occupation_lits"], label="Historique (observé)", color="blue", linewidth=2)
    ax1.plot(pred_stock["date"], pred_stock["occupation_lits_pred"], label=f"Prévision {horizon_test}j (stock-flux)", color="red", linewidth=2)
    
    if "occupation_lits_low" in pred_stock.columns and "occupation_lits_high" in pred_stock.columns:
        ax1.fill_between(
            pred_stock["date"],
            pred_stock["occupation_lits_low"],
            pred_stock["occupation_lits_high"],
            alpha=0.2,
            color="red",
            label="IC 95%"
        )
    
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Lits occupés")
    ax1.set_title(f"Validation : Occupation des lits (historique + prévision {horizon_test}j)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axvline(hist["date"].iloc[-1], color="gray", linestyle="--", alpha=0.5, label="Fin historique")
    
    # Graphique 2 : Taux d'occupation
    ax2 = axes[1]
    ax2.plot(hist["date"], hist["taux_occupation"], label="Historique", color="blue", linewidth=2)
    taux_pred = pred_stock["occupation_lits_pred"] / 1800
    ax2.plot(pred_stock["date"], taux_pred, label=f"Prévision {horizon_test}j", color="red", linewidth=2)
    ax2.axhline(0.85, color="orange", linestyle="--", label="Seuil alerte (85%)")
    ax2.axhline(0.95, color="darkred", linestyle="--", label="Seuil critique (95%)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Taux d'occupation")
    ax2.set_title("Validation : Taux d'occupation (seuils d'alerte)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axvline(hist["date"].iloc[-1], color="gray", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    output_path = ROOT / "tests" / "validation_correction_stock.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Graphiques sauvegardés : {output_path}")
    
    # Ne pas afficher si pas d'environnement graphique
    # plt.show()
else:
    print("❌ Impossible de générer les graphiques (prédiction vide)")

# 4. Résumé final
print(f"\n[4/4] Résumé de la validation")
print("=" * 70)

# Relancer un test complet
pred_14j = predict_occupation_from_admissions(occupation_df, horizon_jours=14, use_best_admissions=True)

if not pred_14j.empty:
    occ_vals = pred_14j["occupation_lits_pred"].values
    variations = np.diff(occ_vals)
    pct_variations = np.abs(variations / occ_vals[:-1]) * 100
    faible_variation = (pct_variations < 0.5).sum() / len(pct_variations) * 100
    
    print(f"\n📊 Statistiques sur prévision 14 jours :")
    print(f"   - Occupation min : {occ_vals.min():.0f} lits")
    print(f"   - Occupation max : {occ_vals.max():.0f} lits")
    print(f"   - Occupation moy : {occ_vals.mean():.0f} lits")
    print(f"   - Amplitude : {occ_vals.max() - occ_vals.min():.0f} lits")
    print(f"   - Jours avec variations >0.5% : {100-faible_variation:.0f}%")
    
    if faible_variation < 50:
        print("\n✅ SUCCÈS : Le modèle génère des prévisions dynamiques (pas de patterns répétitifs)")
        print("✅ La correction du modèle stock-flux est VALIDÉE")
    else:
        print("\n⚠️  ATTENTION : Beaucoup de jours ont des variations faibles")
        print("   Vérifier visuellement les graphiques générés")
else:
    print("\n❌ Échec de la validation")

print("\n" + "=" * 70)
print("FIN DU TEST")
print("=" * 70)
