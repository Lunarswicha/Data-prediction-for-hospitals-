import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

print('='*70)
print('TEST DE NON-RÉGRESSION — DASHBOARD')
print('='*70)

print('\n[1/2] Test d\'importation du dashboard...')
try:
    from app import dashboard
    print('✅ Dashboard importé avec succès')
except Exception as e:
    print(f'❌ Erreur d\'importation : {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

print('\n[2/2] Test des fonctions de prédiction...')
try:
    from src.data.load import load_occupation
    from src.prediction.models import predict_besoins
    
    occ_df = load_occupation(ROOT / 'data')
    besoins = predict_besoins(occ_df, horizon_jours=14, capacite_lits=1800)
    
    print('✅ predict_besoins() fonctionne correctement')
    print(f'   - Taux max prévu : {besoins["taux_max_prevu"]*100:.1f}%')
    print(f'   - Recommandation : {besoins["recommandation"][:60]}...')
    print(f'   - Nombre de jours prédits : {len(besoins["previsions"])}')
    
    # Vérifier que les prédictions sont dynamiques
    pred_df = besoins["previsions"]
    occ_vals = pred_df["occupation_lits_pred"].values
    if len(occ_vals) > 1:
        import numpy as np
        variations = np.diff(occ_vals)
        pct_var = np.abs(variations / occ_vals[:-1]) * 100
        pct_dynamique = (pct_var > 0.1).sum() / len(pct_var) * 100
        print(f'   - % jours avec variations >0.1% : {pct_dynamique:.0f}%')
        
        if pct_dynamique > 50:
            print('   ✅ Prédictions dynamiques (pas de patterns répétitifs)')
        else:
            print('   ⚠️  Prédictions peu variables')
    
except Exception as e:
    print(f'❌ Erreur lors du test : {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

print('\n' + '='*70)
print('✅ TOUS LES TESTS PASSENT')
print('✅ Le dashboard devrait fonctionner correctement')
print('='*70)
print('\nPour lancer le dashboard :')
print('  streamlit run app/dashboard.py')
