import io
import sqlite3
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from model import connect, import_games, features, FEATURES

st.set_page_config(page_title='NBA Predictor', page_icon='🏀', layout='centered')
st.title('🏀 NBA Predictor')
st.caption('Laboratoire de prédictions NBA • saison régulière • aucune donnée fictive')
st.info('Cette version gratuite fonctionne avec un CSV de matchs vérifiés. Elle ne télécharge pas automatiquement les résultats, les cotes ou les blessures et ne garantit pas une exécution automatique à 9 h ou 22 h.')

with st.expander('Format du fichier CSV attendu'):
    st.code('game_id,season,tipoff_utc,home,away,home_pts,away_pts\n',language='csv')
    st.write('Exemple de structure uniquement (sans matchs inventés). Saison : 2018-19, 2019-20, etc. Heure de début avec fuseau UTC, ex. 2026-11-16T00:30:00+00:00. Scores vides pour matchs à venir. Identifiants d’équipes cohérents.')
file=st.file_uploader('Importer un CSV de saison régulière',type=['csv'])
if file is None:
    st.warning('Aucune donnée chargée : aucune prédiction ni performance ne peut être calculée.')
    st.stop()
try:
    raw=file.getvalue()
    conn=connect(':memory:')
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix='.csv',delete=False) as temp:
        temp.write(raw); path=temp.name
    try:
        import_games(conn,path)
    finally:
        os.unlink(path)
    df=features(conn)
    st.success(f'{len(df):,} rencontres importées et vérifiées structurellement.'.replace(',',' '))
except Exception as e:
    st.error(f'Import impossible : {e}')
    st.stop()

st.subheader('Validation chronologique')
completed=df[df.target.notna()].copy()
completed['target']=completed.target.astype(int)
seasons=sorted(completed.season.unique())
if len(seasons)<2:
    st.warning('Au moins deux saisons avec des résultats sont nécessaires pour une évaluation chronologique.')
else:
    test_season=st.selectbox('Première saison de test',seasons[1:],index=max(0,len(seasons)-3))
    if st.button('Lancer le backtest',type='primary'):
        results=[]
        for season in seasons:
            if season<test_season:continue
            tr=completed[completed.season<season]
            te=completed[completed.season==season]
            if len(tr)<100 or len(te)<20 or tr.target.nunique()<2:continue
            mdl=LogisticRegression(max_iter=1000).fit(tr[FEATURES],tr.target)
            for name,p in [('Logistique',mdl.predict_proba(te[FEATURES])[:,1]),('Elo',te.elo_prob.to_numpy())]:
                results.append({'Saison':season,'Modèle':name,'Matchs':len(te),'Brier':brier_score_loss(te.target,p),'Log loss':log_loss(te.target,p,labels=[0,1]),'Précision':accuracy_score(te.target,p>=.5)})
        if results:
            report=pd.DataFrame(results)
            st.dataframe(report,hide_index=True,use_container_width=True)
            st.download_button('Télécharger le bilan CSV',report.to_csv(index=False).encode('utf-8'),'backtest_nba.csv','text/csv')
        else:
            st.warning('Historique insuffisant pour un backtest valide.')

st.subheader('Prévisions de la prochaine nuit (heure française)')
st.caption('À 22 h, le rapport vise les matchs commençant à partir de minuit le lendemain, jusqu’à 23 h 59 (heure de Paris). Les calculs affichés ici sont faits à l’ouverture : ils ne constituent PAS des snapshots automatiques de 9 h et 22 h.')
paris_now=pd.Timestamp.now(tz='Europe/Paris')
report_day=paris_now.date() + pd.Timedelta(days=1)
report_day=st.date_input('Date des matchs (heure de Paris)',value=report_day)
st.subheader('Matchs à venir : estimation exploratoire')
st.caption('Calculée au moment où tu ouvres la page, et non une prédiction archivée à 9 h ou T−60. Sans archives de disponibilité, elle ne doit pas être considérée comme une simulation pré-match historique.')
upcoming=df[df.target.isna()].copy()
if upcoming.empty:
    st.write('Aucun match à venir dans le CSV.')
elif len(completed)<100 or completed.target.nunique()<2:
    st.warning('Au moins 100 matchs terminés sont requis pour entraîner le modèle.')
else:
    now=pd.Timestamp.now(tz='UTC')
    upcoming=upcoming[(pd.to_datetime(upcoming.tipoff_utc,utc=True)>now) & (pd.to_datetime(upcoming.tipoff_utc,utc=True).dt.tz_convert('Europe/Paris').dt.date==report_day)]
    if upcoming.empty:
        st.write('Aucun match futur dans le CSV.')
    else:
        # Recompute using only results from games whose tipoff has passed. Publication-time guarantees require a richer feed.
        safe=connect(':memory:')
        for r in pd.read_sql_query('SELECT * FROM games',conn).itertuples(index=False):
            known=pd.Timestamp(r.tipoff_utc)<now
            safe.execute('INSERT INTO games VALUES(?,?,?,?,?,?,?)',(r.game_id,r.season,r.tipoff_utc,r.home,r.away,r.home_pts if known else None,r.away_pts if known else None))
        safe.commit()
        frozen=features(safe)
        training=frozen[(frozen.target.notna()) & (pd.to_datetime(frozen.tipoff_utc,utc=True)<now)].copy()
        if len(training)<100 or training.target.nunique()<2:
            st.warning('Pas assez de résultats antérieurs pour entraîner le modèle.')
        else:
            mdl=LogisticRegression(max_iter=1000).fit(training[FEATURES],training.target.astype(int))
            candidates=frozen[(frozen.target.isna()) & (pd.to_datetime(frozen.tipoff_utc,utc=True)>now) & (pd.to_datetime(frozen.tipoff_utc,utc=True).dt.tz_convert('Europe/Paris').dt.date==report_day)].copy()
            if candidates.empty:
                st.info('Aucun match futur disponible dans le CSV pour cette date.')
                st.stop()
            # Two separate score regressions trained only on completed games before now.
            scores=pd.read_sql_query('SELECT game_id,home_pts,away_pts FROM games',safe)
            train_scores=training.merge(scores,on='game_id',how='left')
            if train_scores[['home_pts','away_pts']].isna().any().any():
                st.error('Scores incomplets dans les données d’entraînement.')
                st.stop()
            home_model=Ridge(alpha=100.0).fit(train_scores[FEATURES],train_scores.home_pts)
            away_model=Ridge(alpha=100.0).fit(train_scores[FEATURES],train_scores.away_pts)
            candidates['Points domicile']=home_model.predict(candidates[FEATURES]).round(1)
            candidates['Points extérieur']=away_model.predict(candidates[FEATURES]).round(1)
            candidates['Total points']=candidates['Points domicile']+candidates['Points extérieur']
            candidates['Écart domicile']=candidates['Points domicile']-candidates['Points extérieur']
            candidates['Probabilité domicile']=mdl.predict_proba(candidates[FEATURES])[:,1]
            candidates['Probabilité extérieur']=1-candidates['Probabilité domicile']
            candidates['Heure Paris']=pd.to_datetime(candidates.tipoff_utc,utc=True).dt.tz_convert('Europe/Paris').dt.strftime('%d/%m/%Y %H:%M')
            view=candidates[['Heure Paris','home','away','Points domicile','Points extérieur','Total points','Écart domicile','Probabilité domicile','Probabilité extérieur']].rename(columns={'home':'Domicile','away':'Extérieur'})
            st.dataframe(view,hide_index=True,use_container_width=True,column_config={'Probabilité domicile':st.column_config.ProgressColumn('Probabilité domicile',min_value=0,max_value=1,format='%.1f'),'Probabilité extérieur':st.column_config.ProgressColumn('Probabilité extérieur',min_value=0,max_value=1,format='%.1f')})
            st.download_button('Exporter les estimations',view.to_csv(index=False).encode('utf-8'),'estimations_nba.csv','text/csv')

st.divider()
st.caption('Les scores sont des estimations Ridge exploratoires, non calibrées et sans validation de leur erreur absolue. Les probabilités de victoire proviennent d’un modèle distinct et peuvent donc ne pas être cohérentes avec les scores. Version de démonstration analytique. Les imports ne sont pas conservés durablement sur Streamlit Community Cloud. Aucune cote ni conseil de mise.')
