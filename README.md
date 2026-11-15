# NBA Predictor — Streamlit iPhone

Application web mobile pour charger un historique NBA réel, effectuer un backtest chronologique Elo vs régression logistique et calculer des estimations exploratoires pour les matchs futurs présents dans le fichier.

## Déploiement sans programmation (GitHub + Streamlit Community Cloud)
1. Créez un compte GitHub sur https://github.com et connectez-vous.
2. Créez un nouveau dépôt **privé** ou public `nba-predictor` (le déploiement Streamlit peut demander des autorisations d'accès).
3. Sur la page du dépôt, utilisez **Add file > Upload files** et envoyez **app.py**, **model.py**, **requirements.txt** (pas le ZIP lui-même). Validez avec **Commit changes**.
4. Connectez-vous à https://share.streamlit.io avec GitHub, puis choisissez **Create app**, le dépôt et la branche, et le fichier principal `app.py`. Lancez **Deploy**.
5. Ouvrez l'URL générée dans Safari sur iPhone, puis **Partager > Sur l'écran d'accueil**.
6. Importez un fichier CSV réel de saison régulière dans l'interface. Colonnes : `game_id,season,tipoff_utc,home,away,home_pts,away_pts`. `tipoff_utc` doit inclure le fuseau horaire ; scores vides pour matchs à venir. Utilisez des IDs d'équipes cohérents et des saisons du type `2018-19`.
7. Lancez **Lancer le backtest** pour afficher les performances. Le CSV n'est pas fourni : ne jamais fabriquer de résultats.

## Limites importantes
- Le service gratuit peut s'endormir ; l'application ne se met pas à jour toute seule et n'est pas un ordonnanceur fiable.
- Les imports en mémoire ne persistent pas d'une session à l'autre.
- La version actuelle **n'effectue pas** de prédictions horodatées automatiquement à 09:00 ni à T-60. Pour cela, il faut un service d'ordonnancement, une source de calendrier fiable, une base persistante et des snapshots de données avec dates de disponibilité.
- Les blessures officielles ne sont pas intégrées au modèle, et les cotes ne sont pas récupérées.
- Les backtests historiques évaluent les variables Elo/repos, pas les snapshots de blessures ou cotes aux deux horaires.
- Les estimations en direct ne prouvent pas à elles seules que tous les résultats utilisés étaient publiés au moment du calcul ; une source avec timestamps de publication est nécessaire pour une garantie stricte.
- Aucun rendement financier n'est garanti.

## Localement (facultatif)
`pip install -r requirements.txt` puis `streamlit run app.py`.


## Version 2 : points et horaires France

L’application propose maintenant une estimation exploratoire des points domicile et extérieur (deux régressions Ridge entraînées sur les matchs terminés), le total et l’écart. La probabilité de victoire reste estimée séparément : il n’y a pas encore de modèle de scores joint ni de calibration validée.

Le filtre des rencontres suit la date locale Europe/Paris : pour un rapport du 15 novembre à 22 h, sélectionner le 16 novembre (00 h 00–23 h 59). Le changement d’heure est géré par le fuseau Europe/Paris.

IMPORTANT : cette version ne récupère toujours pas automatiquement le calendrier, les blessures ou les cotes. Elle ne produit pas automatiquement des snapshots archivés à 9 h et 22 h. Il faut un service de collecte/planification et un stockage persistant pour cette fonctionnalité. Ne pas utiliser les estimations comme des résultats backtestés.
