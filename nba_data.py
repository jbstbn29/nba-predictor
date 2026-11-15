"""NBA.com ingestion. No guessed tipoff timestamps or fabricated scores."""
import time
import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog, ScheduleLeagueV2

COLUMNS=['game_id','season','tipoff_utc','home','away','home_pts','away_pts']

def season_label(year):
    return f'{year}-{str(year+1)[-2:]}'

def fetch_season(year):
    season=season_label(year)
    schedule=ScheduleLeagueV2(season=season).season_games.get_data_frame()
    logs=LeagueGameLog(season=season,season_type_all_star='Regular Season',player_or_team_abbreviation='T').league_game_log.get_data_frame()
    if schedule.empty or logs.empty:
        raise ValueError(f'{season}: calendrier ou résultats indisponibles')
    # NBA schedule columns vary by endpoint release: require actual UTC kickoff and identifiers.
    cols={str(c).lower():c for c in schedule.columns}
    def col(*names):
        for name in names:
            if name.lower() in cols:return cols[name.lower()]
        raise ValueError(f'{season}: colonne manquante parmi {names}. Colonnes reçues: {list(schedule.columns)}')
    gid=col('gameId','GAME_ID')
    when=col('gameDateTimeUTC','GAME_DATE_TIME_UTC','gameDateTimeUtc')
    homeid=col('homeTeam_teamId','homeTeamId','homeTeamID','HOME_TEAM_ID')
    awayid=col('awayTeam_teamId','awayTeamId','awayTeamID','AWAY_TEAM_ID')
    # Some versions expose nested team dicts instead of flattened columns.
    s=schedule.copy()
    s['game_id']=s[gid].astype(str).str.zfill(10)
    s['tipoff_utc']=pd.to_datetime(s[when],utc=True,errors='coerce')
    s['home_id']=pd.to_numeric(s[homeid],errors='coerce')
    s['away_id']=pd.to_numeric(s[awayid],errors='coerce')
    logs=logs.copy()
    logs['GAME_ID']=logs.GAME_ID.astype(str).str.zfill(10)
    logs['TEAM_ID']=pd.to_numeric(logs.TEAM_ID,errors='coerce')
    logs['PTS']=pd.to_numeric(logs.PTS,errors='coerce')
    logs=logs.drop_duplicates(['GAME_ID','TEAM_ID'])
    home=logs[['GAME_ID','TEAM_ID','TEAM_ABBREVIATION','PTS']].rename(columns={'TEAM_ID':'home_id','TEAM_ABBREVIATION':'home','PTS':'home_pts'})
    away=logs[['GAME_ID','TEAM_ID','TEAM_ABBREVIATION','PTS']].rename(columns={'TEAM_ID':'away_id','TEAM_ABBREVIATION':'away','PTS':'away_pts'})
    s=s.merge(home,left_on=['game_id','home_id'],right_on=['GAME_ID','home_id'],how='left').drop(columns=['GAME_ID'])
    s=s.merge(away,left_on=['game_id','away_id'],right_on=['GAME_ID','away_id'],how='left').drop(columns=['GAME_ID'])
    # For future fixtures, derive team abbreviations from official NBA team IDs.
    from nba_api.stats.static import teams
    id_to_abbr={t['id']:t['abbreviation'] for t in teams.get_teams()}
    s['home']=s['home'].fillna(s.home_id.map(id_to_abbr))
    s['away']=s['away'].fillna(s.away_id.map(id_to_abbr))
    # Strictly regular-season game IDs; exclude postponed/cancelled fixtures without timestamps.
    s=s[s.game_id.str.startswith('002')].copy()
    s['season']=season
    s=s.dropna(subset=['tipoff_utc','home','away'])
    s=s[s.home!=s.away]
    s['tipoff_utc']=s.tipoff_utc.dt.strftime('%Y-%m-%dT%H:%M:%S+00:00')
    # Games with only one reported score are treated as unplayed until data complete.
    partial=s.home_pts.isna()!=s.away_pts.isna()
    s.loc[partial,['home_pts','away_pts']]=float('nan')
    result=s[COLUMNS].drop_duplicates('game_id')
    if result.empty:raise ValueError(f'{season}: aucun match valide')
    return result

def fetch_all(start=2018,end=2026,progress=None):
    frames=[];errors=[]
    for year in range(start,end+1):
        try:
            frames.append(fetch_season(year))
        except Exception as exc:
            errors.append(f'{season_label(year)} : {exc}')
        if progress:progress(year,errors)
        time.sleep(0.7)
    if not frames:raise RuntimeError('Aucune saison récupérée. '+ ' | '.join(errors))
    return pd.concat(frames,ignore_index=True).drop_duplicates('game_id'),errors
