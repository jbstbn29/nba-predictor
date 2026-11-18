import io
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
from model import connect,import_games,features,FEATURES

st.set_page_config(page_title='NBA Predictor',page_icon='🏀',layout='centered')
st.title('🏀 NBA Predictor')
st.caption('Saison régulière • données NBA.com • fuseau Europe/Paris')
st.info('Les données sont téléchargées à la demande. Pas encore de captures automatiques à 9 h et 22 h, ni de blessures/cotes intégrées.')

if 'csv' not in st.session_state:st.session_state.csv=None
with st.expander('Récupérer les données via ESPN (alternative à NBA.com)',expanded=st.session_state.csv is None):
    st.write('Source ESPN non officielle : télécharger une saison à la fois. Les données ne sont pas inventées. Le service peut être indisponible ou limiter les requêtes.')
    season_year=st.selectbox('Saison NBA',list(range(2018,2027)),index=7,format_func=lambda y:f'{y}-{str(y+1)[-2:]}')
    if st.button('📥 Récupérer cette saison',type='primary'):
        from espn_data import fetch_season
        bar=st.progress(0)
        try:
            data,errors=fetch_season(season_year,lambda i,n:bar.progress(min(i/n,1.0)))
            if errors:
                st.error(f'{len(errors)} périodes non récupérées. Saison incomplète : export désactivé pour éviter un historique trompeur.')
                st.code('\n'.join(errors[:5]))
            else:
                st.session_state.csv=data.to_csv(index=False).encode('utf-8')
                st.success(f'{len(data)} rencontres chargées pour {season_year}-{str(season_year+1)[-2:]}.')
        except Exception as exc:st.error('Source ESPN indisponible : '+str(exc))
    if st.session_state.csv:
        st.download_button('Sauvegarder le CSV',st.session_state.csv,'nba_regular_season.csv','text/csv')
    st.caption('Pour couvrir toutes les saisons, télécharge chaque saison puis fusionne les CSV hors de cette première interface. Une seule saison ne suffit pas au backtesting multi-saisons.')
file=st.file_uploader('Ou importer un CSV déjà téléchargé',type='csv')
raw=file.getvalue() if file is not None else st.session_state.csv
if raw is None:
    st.warning('Aucune donnée disponible. Lance la récupération ci-dessus ou importe un CSV.')
    st.stop()
try:
    from tempfile import NamedTemporaryFile
    import os
    conn=connect(':memory:')
    with NamedTemporaryFile(suffix='.csv',delete=False) as tmp:
        tmp.write(raw);path=tmp.name
    try:import_games(conn,path)
    finally:os.unlink(path)
    df=features(conn)
except Exception as e:
    st.error(f'Impossible de traiter les données : {e}');st.stop()
completed=df[df.target.notna()].copy()
completed['target']=completed.target.astype(int)
st.metric('Matchs terminés disponibles',len(completed))
seasons=sorted(completed.season.unique())
st.subheader('Validation chronologique')
if len(seasons)>1:
    first=st.selectbox('Première saison de test',seasons[1:],index=max(0,len(seasons)-3))
    if st.button('Calculer le backtest'):
        output=[]
        for season in seasons:
            if season<first:continue
            tr=completed[completed.season<season];te=completed[completed.season==season]
            if len(tr)<100 or len(te)<20 or tr.target.nunique()<2:continue
            classifier=LogisticRegression(max_iter=1000).fit(tr[FEATURES],tr.target)
            home_score=Ridge(alpha=100).fit(tr[FEATURES],tr.merge(pd.read_sql_query('SELECT game_id,home_pts FROM games',conn),on='game_id').home_pts)
            away_score=Ridge(alpha=100).fit(tr[FEATURES],tr.merge(pd.read_sql_query('SELECT game_id,away_pts FROM games',conn),on='game_id').away_pts)
            p=classifier.predict_proba(te[FEATURES])[:,1]
            actual=te.merge(pd.read_sql_query('SELECT game_id,home_pts,away_pts FROM games',conn),on='game_id')
            output.append({'Saison':season,'Matchs':len(te),'Brier':round(brier_score_loss(te.target,p),4),'Log loss':round(log_loss(te.target,p),4),'Erreur points domicile':round(mean_absolute_error(actual.home_pts,home_score.predict(te[FEATURES])),2),'Erreur points extérieur':round(mean_absolute_error(actual.away_pts,away_score.predict(te[FEATURES])),2)})
        if output:
            st.dataframe(pd.DataFrame(output),hide_index=True)
            st.download_button('Télécharger le backtest',pd.DataFrame(output).to_csv(index=False),'nba_backtest.csv')
        else:st.warning('Pas assez de saisons valides.')
else:st.warning('Deux saisons terminées au minimum sont nécessaires.')

st.subheader('Prévisions des matchs à venir')
st.caption('Prévisions calculées lors de la consultation, non enregistrées à 9 h / 22 h. Les scores historiques sont utilisés uniquement pour les rencontres déjà commencées ; un résultat récent non encore publié peut manquer.')
now=pd.Timestamp.now(tz='UTC')
paris=now.tz_convert('Europe/Paris')
report_day=st.date_input('Date des rencontres (heure française)',value=paris.date()+pd.Timedelta(days=1))
future=df[(df.target.isna()) & (pd.to_datetime(df.tipoff_utc,utc=True)>now)].copy()
if not future.empty:
    future=future[pd.to_datetime(future.tipoff_utc,utc=True).dt.tz_convert('Europe/Paris').dt.date==report_day]
if future.empty:st.info('Aucune rencontre future disponible pour cette date dans les données téléchargées.')
elif len(completed)<100 or completed.target.nunique()<2:st.warning('Historique insuffisant pour produire des prévisions.')
else:
    # Exclude games whose results were not available before the prediction time.
    training=completed[pd.to_datetime(completed.tipoff_utc,utc=True)<now]
    if len(training)<100:st.warning('Historique insuffisant.')
    else:
        clf=LogisticRegression(max_iter=1000).fit(training[FEATURES],training.target)
        scores=training.merge(pd.read_sql_query('SELECT game_id,home_pts,away_pts FROM games',conn),on='game_id')
        hm=Ridge(alpha=100).fit(scores[FEATURES],scores.home_pts)
        am=Ridge(alpha=100).fit(scores[FEATURES],scores.away_pts)
        result=future[['home','away','tipoff_utc']].copy()
        result['Heure Paris']=pd.to_datetime(result.tipoff_utc,utc=True).dt.tz_convert('Europe/Paris').dt.strftime('%d/%m %H:%M')
        result['Points domicile']=hm.predict(future[FEATURES]).round(1)
        result['Points extérieur']=am.predict(future[FEATURES]).round(1)
        result['Total']=result['Points domicile']+result['Points extérieur']
        result['Probabilité domicile (%)']=(100*clf.predict_proba(future[FEATURES])[:,1]).round(1)
        st.dataframe(result[['Heure Paris','home','away','Points domicile','Points extérieur','Total','Probabilité domicile (%)']],hide_index=True)
        st.caption('Estimations exploratoires : les modèles de scores et de victoire sont indépendants et ne sont pas calibrés ensemble.')
