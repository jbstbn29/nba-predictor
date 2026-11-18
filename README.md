# NBA Predictor V4 — source ESPN

Remplacez les fichiers à la racine du dépôt GitHub, branche `main`, puis laissez Streamlit redéployer. La V4 remplace la récupération NBA.com (timeout) par une récupération saison par saison via l'endpoint public ESPN scoreboard.

Le service ESPN n'est pas une API officielle garantie. Les résultats ne sont pas intégrés dans cette archive : la récupération réseau doit être testée sur Streamlit. En cas de périodes manquantes, l'export est désactivé. Une saison seule ne suffit pas pour valider un modèle multi-saisons.

Les prédictions 9 h / 22 h ne sont pas automatisées ; les blessures, cotes et captures horodatées restent à implémenter. Ne pas interpréter les estimations comme validées ou comme des conseils de mise.
