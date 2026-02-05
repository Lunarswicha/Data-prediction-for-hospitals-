# Configuration — Pitié-Salpêtrière (valeurs indicatives, fictives)
# Référence : ~100 000 passages urgences/an, ~1800 lits

from pathlib import Path

# Chemins
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_GENERATED = PROJECT_ROOT / "data" / "generated"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Capacités fictives (Pitié-Salpêtrière)
CAPACITE_LITS_TOTALE = 1800
CAPACITE_URGENCES_QUOTIDIENNE = 350  # passages/jour en moyenne
SERVICES = [
    "urgences",
    "cardiologie",
    "neurologie",
    "maladies_infectieuses",
    "reanimation",
    "pediatrie",
    "chirurgie",
    "medecine_generale",
]

# Répartition indicative des lits par service (%)
REPARTITION_LITS = {
    "urgences": 0.08,
    "cardiologie": 0.12,
    "neurologie": 0.10,
    "maladies_infectieuses": 0.08,
    "reanimation": 0.06,
    "pediatrie": 0.10,
    "chirurgie": 0.22,
    "medecine_generale": 0.24,
}

# Personnel fictif (ETP indicatif)
PERSONNEL_MEDICAL_ETP = 1200
PERSONNEL_SOIGNANT_ETP = 3500
RATIO_LIT_INFIRMIER = 0.4  # ETP infirmier par lit (indicatif)

# Seuils d'alerte (taux d'occupation)
SEUIL_ALERTE_OCCUPATION = 0.85  # 85 %
SEUIL_CRITIQUE_OCCUPATION = 0.95  # 95 %

# Scénarios de simulation (Pitié-Salpêtrière, Paris)
# Calibrés pour démo : base déjà ~100 % → scénario doit rester dans 115–125 % max (multiplicateurs 1.05–1.10)
# mois_saison : mois (1-12) où le scénario est réaliste ; décalage auto si besoin
SCENARIOS = {
    "epidemie_grippe": {
        "label": "Épidémie grippe (hiver)",
        "surplus_admissions": 0.08,  # mult. pic ~1.08 (était 0.18 → 179 %)
        "services_impactes": ["urgences", "pediatrie", "reanimation"],
        "duree_jours": 45,
        "pic_jour": 15,
        "mois_saison": (11, 12, 1, 2, 3),  # nov.–mars (hiver)
    },
    "greve": {
        "label": "Grève (réduction personnel)",
        "reduction_personnel": 0.08,   # était 0.25
        "reduction_capacite_lits": 0.05,  # était 0.12
        "duree_jours": 14,
        # pas de mois_saison : toute l'année
    },
    "canicule": {
        "label": "Canicule (été)",
        "surplus_admissions": 0.04,  # mult. pic ~1.04 (était 0.08)
        "services_impactes": ["urgences", "reanimation", "medecine_generale"],
        "duree_jours": 21,
        "pic_jour": 10,
        "mois_saison": (6, 7, 8),  # juin–août (été, Île-de-France)
    },
    "afflux_massif": {
        "label": "Afflux massif (événement exceptionnel)",
        "surplus_admissions": 0.10,  # mult. pic ~1.10 (était 0.35)
        "duree_jours": 3,
        "pic_jour": 1,
        # pas de mois_saison : imprévisible
    },
}
