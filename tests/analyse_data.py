import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.load import load_occupation

df = load_occupation(ROOT / 'data')
print('=== Analyse des données historiques ===')
print(f'Occupation moyenne : {df["occupation_lits"].mean():.0f} lits')
print(f'Occupation médiane : {df["occupation_lits"].median():.0f} lits')
print(f'Occupation min : {df["occupation_lits"].min():.0f} lits')
print(f'Occupation max : {df["occupation_lits"].max():.0f} lits')
print(f'\nAdmissions moyennes : {df["admissions_jour"].mean():.0f} par jour')
print(f'Admissions médianes : {df["admissions_jour"].median():.0f} par jour')
print(f'\nTaux d\'occupation moyen : {df["taux_occupation"].mean()*100:.1f}%')
print(f'Taux d\'occupation médian : {df["taux_occupation"].median()*100:.1f}%')
print(f'\nDerniers 30 jours :')
tail = df.tail(30)
print(f'  Occupation moyenne : {tail["occupation_lits"].mean():.0f} lits')
print(f'  Admissions moyennes : {tail["admissions_jour"].mean():.0f} par jour')
print(f'  Dernier jour : {tail["occupation_lits"].iloc[-1]:.0f} lits')

# Calculer la DMS implicite en régime permanent
# En régime permanent : Stock = Admissions × DMS
# Donc DMS = Stock / Admissions
dms_implicite = df["occupation_lits"].mean() / df["admissions_jour"].mean()
print(f'\nDMS implicite (Stock moyen / Admissions moyennes) : {dms_implicite:.2f} jours')
