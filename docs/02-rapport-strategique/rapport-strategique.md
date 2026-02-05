# Rapport stratégique et marketing

**Projet Data — MVP Pitié-Salpêtrière — Promo 2026**

*Freins et leviers d'adoption, recommandations, impact sur la gestion hospitalière, comparaison avec l'existant, perspectives.*

---

## 1. Contexte et positionnement

### 1.1 Besoin de la direction

La direction de l'Hôpital Pitié-Salpêtrière est confrontée à une **variabilité importante des flux** (urgences, admissions, occupation des lits) : pics hivernaux (grippe, bronchiolite), événements exceptionnels (canicule, afflux massif), mouvements sociaux (grèves). Anticiper ces pics permet de **mieux répartir les ressources** (personnel, lits, matériel) et de limiter la saturation des services. Le MVP livré répond à ce besoin en proposant un **tableau de bord de prévision** et de **simulation de scénarios**, assorti de **recommandations automatiques** (alertes, actions).

### 1.2 Rôle de l'outil prédictif dans la gestion

L'outil ne remplace pas la décision humaine ; il **éclaire** la direction et les pôles avec des **prévisions** (admissions, occupation, taux) et des **fourchettes d'incertitude** (IC 95 %). En cas de dépassement des seuils (85 % alerte, 95 % critique), des recommandations sont proposées (renforts, report d'activité, vigilance stocks). La **simulation de scénarios** (épidémie, grève, canicule, afflux massif) permet d'explorer des hypothèses « et si » avant qu'elles ne surviennent. Ce positionnement est aligné avec la littérature : Batal et al. montrent qu'adapter le planning du personnel à la prédiction du flux réduit les départs sans soins (−18,5 %) et les plaintes (−30 %).

---

## 2. Freins et leviers d'adoption

### 2.1 Freins à l'usage d'un outil prédictif en milieu hospitalier

| Type | Exemples |
|------|----------|
| **Organisationnels** | Priorité donnée au court terme (gestion au jour le jour) ; réunions de pilotage peu tournées vers la prévision ; cloisonnement entre services et direction. |
| **Techniques** | Méconnaissance des modèles (« boîte noire ») ; crainte de dépendre à un outil dont la maintenance ou la mise à jour est incertaine ; absence d'intégration avec le SI hospitalier (données réelles). |
| **Culturels** | Habitude de s'appuyer sur l'expérience et l'intuition ; méfiance à l'égard des « chiffres » ou des algorithmes ; peur que l'outil remplace le jugement clinique ou managérial. |

Ces freins sont classiques dans les projets data en santé ; les thèses intégrées (Bouteloup, Lequertier) et les retours d'expérience (DREES, établissements) les mentionnent explicitement.

### 2.2 Leviers pour favoriser l'adoption

- **Formation et communication** : expliquer en termes simples ce que prédisent les modèles (flux, taux d'occupation), comment sont construits les intervalles de confiance et pourquoi la **borne haute** est privilégiée pour la planification (éviter la sous-estimation). Présenter le prototype comme une **aide à la décision**, pas un substitut au pilotage.
- **Pilotage** : intégrer les prévisions et les alertes dans les **réunions de régulation** et de pilotage (indicateurs clés, graphiques, dernières recommandations). Viser une **vue synthétique « direction »** (une page = un écran pour la réunion).
- **Intégration aux process** : définir des règles claires (qui consulte le dashboard, à quelle fréquence ; qui déclenche les actions en cas d'alerte). Prévoir des **exports** (CSV, PDF) pour partager les prévisions et les recommandations avec les pôles.
- **Gestion de la confiance** : documenter les **limites** du MVP (données fictives, validation à faire sur données réelles) et les **métriques de performance** (% à ±10 %, biais) pour que les utilisateurs comprennent le niveau de confiance à accorder aux prévisions.

---

## 3. Recommandations

### 3.1 Amélioration de la perception et de l'efficacité des services

- **Mettre en avant les bénéfices mesurables** : réduction des pics de tension (anticipation), meilleure répartition des effectifs, limitation des départs sans soins et des plaintes (réf. Batal et al.). Communiquer ces objectifs auprès des équipes et de la direction.
- **Associer les utilisateurs** : impliquer les régulateurs, les cadres de pôle et la direction dans la définition des **seuils d'alerte** et des **recommandations** (actions proposées, priorisation). Un outil perçu comme « imposé » sans concertation renforce les freins culturels.
- **Valoriser la complémentarité** : l'outil fournit des **tendances et des fourchettes** ; le jugement humain reste central pour les décisions (report d'activité, renforts, communication interne).

### 3.2 Recommandations concrètes issues du modèle prédictif et des simulations

Les recommandations générées automatiquement par le prototype (module `src/recommendations/`) sont déjà **opérationnelles** : niveau (normal, alerte, critique), message, priorité, liste d'actions (renforts, report d'interventions non urgentes, activation des lits de réserve, alerte régulation). Elles s'appuient sur la **littérature** (Bouteloup : privilégier la borne haute de l'IC ; Batal : adapter le planning à la prédiction). Pour aller plus loin : **personnaliser** les recommandations par service (urgences, réa, etc.) et **exporter** les recommandations (PDF/CSV) pour traçabilité et partage.

---

## 4. Évaluation de l'impact potentiel

### 4.1 Réduction des temps d'attente, optimisation des coûts, répartition des ressources

- **Temps d'attente** : en anticipant les pics, les effectifs peuvent être renforcés aux bonnes périodes ; la littérature (Batal et al.) montre une baisse des départs sans soins et des plaintes lorsque le planning est aligné sur la prédiction. Une évaluation en conditions réelles nécessiterait des indicateurs avant/après (délai moyen aux urgences, taux de départs sans soins).
- **Coûts** : une meilleure répartition des ressources (éviter le surdimensionnement permanent ou, à l'inverse, les crises) peut contribuer à une optimisation des coûts ; une quantification précise demanderait une étude dédiée (coût du personnel, des lits, des déprogrammations).
- **Répartition des ressources** : le tableau de bord et les seuils d'alerte visent explicitement à **ajuster** les ressources en fonction des prévisions ; les scénarios (épidémie, grève, etc.) permettent de tester des hypothèses avant qu'elles ne surviennent.

### 4.2 Indicateurs proposés et méthode d'évaluation

| Indicateur | Description | Méthode |
|------------|-------------|---------|
| Taux d'occupation moyen / max | Niveau d'occupation des lits (réel vs prévu) | Suivi continu ; comparaison prévu/observé sur une période de validation. |
| % de jours à ±10 % (Bouteloup) | Qualité de la prévision (prévu vs observé) | Backtest sur période passée ; sur données réelles, viser 70–85 %. |
| Nombre d'alertes déclenchées / actions engagées | Usage effectif des recommandations | Traçabilité des alertes et des décisions (renforts, reports). |
| Délais de prise en charge, départs sans soins, plaintes | Impact sur la qualité de service | Comparaison avant/après déploiement (données réelles). |

L'évaluation complète suppose un **déploiement en établissement** avec des **données réelles** et un **cadre de pilotage** défini (qui regarde les indicateurs, à quelle fréquence).

---

## 5. Comparaison avec les solutions existantes

### 5.1 Solutions actuelles (national / établissements)

En France, les établissements s'appuient souvent sur :

- **Tableaux de bord d'activité** (tableaux Excel, outils métier) : suivi des flux et de l'occupation en temps réel ou en différé ; peu de **prévision** formalisée.
- **Règles empiriques** : moyennes historiques, « lundi chargé », « hiver tendu » ; peu de modélisation explicite.
- **Projets de recherche ou pilotes** : thèses et études (Bouteloup, Lequertier, Wargon, etc.) sur la prédiction du flux ou de la durée de séjour ; pas toujours déployés en routine.
- **Offres logicielles** : certains éditeurs proposent des modules de prévision ou de planification ; la comparaison détaillée (fonctions, coûts, intégration) dépasse le périmètre de ce rapport.

### 5.2 Axes d'amélioration et différenciation du MVP

Notre MVP se différencie par :

1. **Prévisions avec intervalles de confiance** (IC 95 %) et critère opérationnel **±10 %** (réf. Bouteloup).
2. **Simulation de scénarios** paramétrable (épidémie, grève, canicule, afflux massif ; durée 14–90 j).
3. **Recommandations automatiques** liées aux seuils (85 % / 95 %) et à la borne haute de l'IC.
4. **Modèles multiples** (Holt-Winters, Ridge, SARIMA, Boosting) avec backtest et métriques (MAE, RMSE, % ±10 %, biais).

**Axes d'amélioration** pour se rapprocher ou dépasser les solutions existantes : intégration de **données réelles** (PMSI, RPU) dans un cadre réglementaire ; **données météo** réelles ; **prévision de la durée de séjour** (réf. Lequertier) ; **recommandations par service** ; **planification des lits et du personnel** (ETP, créneaux).

---

## 6. Perspectives d'évolution

### 6.1 Évolutions possibles de l'outil

- **Modèles** : comparaison systématique des modèles (tableau récapitulatif) ; modèle ensembliste ; données météo réelles ; prédiction de la durée de séjour (LOS) pour affiner le modèle stock.
- **Simulation** : scénarios combinés (ex. grève + épidémie), courbes paramétrables, bandes d'incertitude.
- **Alertes et recommandations** : seuils configurables, historique des alertes, recommandations par service, export PDF/CSV.
- **Planification** : effectifs cibles (ETP, renforts), créneaux de déprogrammation, cartographie des lits par service.
- **Données et conformité** : connecteurs sécurisés pour données réelles ; journal des accès ; référentiel HDS si applicable.

### 6.2 Applications futures

- **Extension à d'autres établissements** : le prototype (données fictives) peut servir de démonstrateur ; un déploiement multi-établissements supposerait des paramètres par site (capacité, services, historique) et un cadre réglementaire partagé.
- **Recherche et publication** : reproductibilité des expériences (notebooks, versioning des jeux de données), rapport technique (méthodes, limites, références), éventuelle publication ou communication (conférence, rapport DREES/Santé publique France).
- **Interopérabilité** : échange de prévisions ou d'alertes avec des outils de régulation ou de pilotage (AP-HP, ARS) si les formats et la gouvernance sont définis.

Le document *PISTES-EVOLUTION.md* (dossier 05-reference) détaille ces pistes.

---

*Projet Data Pitié-Salpêtrière — Promo 2026. Rapport stratégique et marketing = 1 document.*
