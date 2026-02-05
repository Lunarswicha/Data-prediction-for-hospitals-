"""Visualisation : Comparaison graphique de tous les modèles."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data.load import load_occupation
from src.prediction.models import (
    predict_holt_winters,
    predict_regression,
    predict_sarima,
    predict_boosting,
    predict_moving_average,
    prepare_series,
)

# Charger les données
occupation_df = load_occupation(ROOT / 'data')
adm = prepare_series(occupation_df, "admissions_jour")

horizon = 60  # 60 jours pour bien voir les différences

print("Génération des prédictions pour tous les modèles...")

# Prédictions
modeles = {
    "Holt-Winters": predict_holt_winters,
    "Régression Ridge": predict_regression,
    "SARIMA": predict_sarima,
    "Boosting XGB": predict_boosting,
    "Moving Average": predict_moving_average,
}

resultats = {}
for nom, fonction in modeles.items():
    print(f"  - {nom}...", end=" ")
    try:
        pred = fonction(adm, horizon_jours=horizon)
        if pred is not None and len(pred) > 0:
            values = pred["prediction"].values if "prediction" in pred.columns else pred.values
            resultats[nom] = values
            print(f"✅ (moyenne = {values.mean():.1f})")
        else:
            print("❌")
    except Exception as e:
        print(f"❌ ({e})")

# Créer le graphique
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Graphique 1 : Toutes les courbes superposées
ax1 = axes[0]
dates = pd.date_range(adm.index[-1] + pd.Timedelta(days=1), periods=horizon, freq='D')

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i, (nom, values) in enumerate(resultats.items()):
    ax1.plot(dates, values, label=nom, linewidth=2, color=colors[i], alpha=0.8)

ax1.set_title("Comparaison des 5 Modèles de Prédiction (60 jours)", fontsize=14, fontweight='bold')
ax1.set_xlabel("Date")
ax1.set_ylabel("Admissions quotidiennes")
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=adm.iloc[-90:].mean(), linestyle='--', color='gray', alpha=0.5, label='Moyenne 90j historique')

# Graphique 2 : Écarts relatifs par rapport à la moyenne des 5 modèles
ax2 = axes[1]

# Calculer la moyenne des 5 modèles
ensemble_mean = np.mean(list(resultats.values()), axis=0)

for i, (nom, values) in enumerate(resultats.items()):
    ecarts_pct = ((values - ensemble_mean) / ensemble_mean) * 100
    ax2.plot(dates, ecarts_pct, label=nom, linewidth=2, color=colors[i], alpha=0.8)

ax2.set_title("Écarts par rapport à la Moyenne Ensemble (%)", fontsize=14, fontweight='bold')
ax2.set_xlabel("Date")
ax2.set_ylabel("Écart relatif (%)")
ax2.axhline(y=0, linestyle='-', color='black', linewidth=1)
ax2.axhline(y=5, linestyle='--', color='red', alpha=0.3)
ax2.axhline(y=-5, linestyle='--', color='red', alpha=0.3)
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.text(dates[5], 5.5, "±5% (seuil de divergence)", fontsize=9, color='red', alpha=0.7)

plt.tight_layout()
output_path = ROOT / 'docs' / 'comparaison_modeles.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Graphique sauvegardé : {output_path}")

# Statistiques finales
print("\n" + "="*80)
print("STATISTIQUES DE DIVERGENCE")
print("="*80)

for nom, values in resultats.items():
    ecart_vs_ensemble = np.abs(values - ensemble_mean).mean()
    ecart_pct = (ecart_vs_ensemble / ensemble_mean.mean()) * 100
    print(f"{nom:20s} : Écart moyen vs ensemble = {ecart_vs_ensemble:.1f} ({ecart_pct:.1f}%)")

# Corrélations
print("\n" + "="*80)
print("MATRICE DE CORRÉLATION")
print("="*80)

noms = list(resultats.keys())
for i in range(len(noms)):
    for j in range(i+1, len(noms)):
        corr = np.corrcoef(resultats[noms[i]], resultats[noms[j]])[0, 1]
        print(f"{noms[i]:20s} vs {noms[j]:20s} : {corr:.3f}")
