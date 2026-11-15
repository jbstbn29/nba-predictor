# NBA Predictor V3

Upload the four files `app.py`, `model.py`, `nba_data.py`, and `requirements.txt` to the root of your existing GitHub repository, replacing the old versions. Streamlit Community Cloud automatically redeploys.

In the app, click **Récupérer les saisons NBA**. Download and keep the generated CSV: Streamlit session state is not persistent. The source is NBA.com via `nba_api` and requests may fail on cloud IP addresses. Failed seasons are reported; no synthetic games are inserted.

This is a prototype. It does not yet automate 09:00/22:00 Paris-time snapshots or integrate injuries/odds. Check chronological data availability before relying on forecasts. NBA.com data are subject to NBA.com terms.
