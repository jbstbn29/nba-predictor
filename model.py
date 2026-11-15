"""NBA pregame baseline. No synthetic data. CSV inputs, UTC ISO timestamps."""
import argparse, json, sqlite3
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

DDL='''CREATE TABLE IF NOT EXISTS games(game_id TEXT PRIMARY KEY,season TEXT NOT NULL,tipoff_utc TEXT NOT NULL,home TEXT NOT NULL,away TEXT NOT NULL,home_pts INTEGER,away_pts INTEGER, CHECK(home != away));
CREATE TABLE IF NOT EXISTS injury_reports(game_id TEXT NOT NULL,player TEXT NOT NULL,team TEXT NOT NULL,status TEXT NOT NULL,report_utc TEXT NOT NULL,source TEXT NOT NULL,PRIMARY KEY(game_id,player,report_utc));
CREATE TABLE IF NOT EXISTS predictions(game_id TEXT NOT NULL,slot TEXT NOT NULL,asof_utc TEXT NOT NULL,prob_home REAL NOT NULL,model TEXT NOT NULL,features_json TEXT NOT NULL,created_utc TEXT NOT NULL,PRIMARY KEY(game_id,slot,asof_utc));'''
FEATURES=['elo_diff','rest_diff','home_b2b','away_b2b']

def utc(s):
    d=pd.Timestamp(s)
    if d.tzinfo is None: raise ValueError('Timezone required for '+str(s))
    return d.tz_convert('UTC')

def connect(db):
    c=sqlite3.connect(db); c.executescript(DDL); return c

def import_games(c,path):
    df=pd.read_csv(path,dtype={'game_id':str,'season':str,'home':str,'away':str})
    required={'game_id','season','tipoff_utc','home','away','home_pts','away_pts'}
    if not required.issubset(df): raise ValueError('Missing '+str(required-set(df)))
    rows=[]
    for r in df.to_dict('records'):
        t=utc(r['tipoff_utc']).isoformat()
        if r['home']==r['away']: raise ValueError('Same teams')
        hp=None if pd.isna(r['home_pts']) else int(r['home_pts'])
        ap=None if pd.isna(r['away_pts']) else int(r['away_pts'])
        if (hp is None)!=(ap is None): raise ValueError('Incomplete score')
        rows.append((str(r['game_id']),str(r['season']),t,r['home'],r['away'],hp,ap))
    c.executemany('INSERT INTO games VALUES(?,?,?,?,?,?,?) ON CONFLICT(game_id) DO UPDATE SET season=excluded.season,tipoff_utc=excluded.tipoff_utc,home=excluded.home,away=excluded.away,home_pts=excluded.home_pts,away_pts=excluded.away_pts',rows);c.commit()
    print('Imported',len(rows),'games')

def import_injuries(c,path):
    df=pd.read_csv(path,dtype=str).fillna('')
    required={'game_id','player','team','status','report_utc','source'}
    if not required.issubset(df):raise ValueError('Missing '+str(required-set(df)))
    rows=[(r['game_id'],r['player'],r['team'],r['status'],utc(r['report_utc']).isoformat(),r['source']) for r in df.to_dict('records')]
    c.executemany('INSERT OR REPLACE INTO injury_reports VALUES(?,?,?,?,?,?)',rows);c.commit();print('Imported',len(rows),'injury observations')

def features(c):
    df=pd.read_sql_query('SELECT * FROM games ORDER BY tipoff_utc,game_id',c)
    if df.empty:raise ValueError('No games imported')
    ratings={}; last={}; rows=[]; i=0
    # Batch by tipoff timestamp: no simultaneous game can inform another.
    for stamp, group in df.groupby('tipoff_utc',sort=True):
        pending=[]
        for r in group.itertuples(index=False):
            h,a=r.home,r.away; rh=ratings.get(h,1500.);ra=ratings.get(a,1500.)
            date=utc(r.tipoff_utc)
            resth=7 if h not in last else max(0,min(7,(date-last[h]).total_seconds()/86400-1))
            resta=7 if a not in last else max(0,min(7,(date-last[a]).total_seconds()/86400-1))
            # ELO prior to game; fixed home advantage 65 Elo points
            p=1/(1+10**((ra-rh-65)/400))
            x={'elo_diff':rh-ra,'rest_diff':resth-resta,'home_b2b':int(resth<0.5),'away_b2b':int(resta<0.5)}
            rows.append({'game_id':r.game_id,'season':r.season,'tipoff_utc':r.tipoff_utc,'home':h,'away':a,'elo_prob':p,**x,'target':None if pd.isna(r.home_pts) else int(r.home_pts>r.away_pts)})
            if not pd.isna(r.home_pts):pending.append((h,a,rh,ra,p,int(r.home_pts>r.away_pts),date))
        for h,a,rh,ra,p,y,date in pending:
            delta=20*(y-p);ratings[h]=rh+delta;ratings[a]=ra-delta;last[h]=date;last[a]=date
    return pd.DataFrame(rows)

def train_eval(c,first_test):
    df=features(c); df=df[df.target.notna()].copy();df['target']=df.target.astype(int)
    seasons=sorted(df.season.unique());print('Seasons:',seasons)
    outputs=[]
    for season in seasons:
        if season<first_test:continue
        tr=df[df.season<season];te=df[df.season==season]
        if len(tr)<100 or len(te)<20 or tr.target.nunique()<2:continue
        model=LogisticRegression(max_iter=1000)
        model.fit(tr[FEATURES],tr.target)
        p=model.predict_proba(te[FEATURES])[:,1]
        for name,probs in [('logistic',p),('elo',te.elo_prob.to_numpy())]:
            outputs.append({'season':season,'model':name,'n':len(te),'brier':round(brier_score_loss(te.target,probs),5),'logloss':round(log_loss(te.target,probs,labels=[0,1]),5),'accuracy':round(accuracy_score(te.target,probs>=.5),5)})
    if not outputs:raise ValueError('Not enough historical training/test data')
    print(pd.DataFrame(outputs).to_string(index=False))

def snapshot(c,slot,day):
    df=features(c)
    tz=ZoneInfo('Europe/Paris'); today=pd.Timestamp(day).date()
    # Daily report is 09:00 Paris. T-60 is one hour before each tipoff.
    today_rows=df[pd.to_datetime(df.tipoff_utc,utc=True).dt.tz_convert(tz).dt.date==today]
    if today_rows.empty:print('No games on selected Paris date');return
    for r in today_rows.itertuples(index=False):
        tip=utc(r.tipoff_utc)
        asof=(pd.Timestamp(datetime.combine(today,datetime.min.time(),tzinfo=tz))+pd.Timedelta(hours=9)).tz_convert('UTC') if slot=='morning' else tip-pd.Timedelta(hours=1)
        if asof>=tip:continue
        if abs((pd.Timestamp.now(tz='UTC')-asof).total_seconds())>300:
            print('SKIP: run within 5 minutes of cutoff',r.game_id);continue
        # Rebuild features using only results known at cutoff, never later scores.
        safe=connect(':memory:')
        historical=pd.read_sql_query('SELECT * FROM games',c)
        for gr in historical.itertuples(index=False):
            known=utc(gr.tipoff_utc)<asof
            safe.execute('INSERT INTO games VALUES(?,?,?,?,?,?,?)',(gr.game_id,gr.season,gr.tipoff_utc,gr.home,gr.away,gr.home_pts if known else None,gr.away_pts if known else None))
        safe.commit()
        frozen=features(safe)
        target=frozen[frozen.game_id==r.game_id].iloc[0]
        x=pd.DataFrame([{f:target[f] for f in FEATURES}])
        tr=frozen[(pd.to_datetime(frozen.tipoff_utc,utc=True)<asof)&frozen.target.notna()]
        if len(tr)<100 or tr.target.nunique()<2:print('SKIP insufficient training',r.game_id);continue
        model=LogisticRegression(max_iter=1000).fit(tr[FEATURES],tr.target.astype(int))
        prob=float(model.predict_proba(x)[0,1])
        # Injury reports stored as snapshots, but not used until impact coefficients are trained on as-of data.
        count=c.execute('SELECT COUNT(*) FROM injury_reports WHERE game_id=? AND report_utc<=?',(r.game_id,asof.isoformat())).fetchone()[0]
        payload={**x.iloc[0].to_dict(),'injury_rows_asof':count,'injury_feature_enabled':False}
        now=pd.Timestamp.now(tz='UTC').isoformat()
        c.execute('INSERT OR IGNORE INTO predictions VALUES(?,?,?,?,?,?,?)',(r.game_id,slot,asof.isoformat(),prob,'logistic_v1',json.dumps(payload),now));c.commit()
        print(r.game_id,slot,asof.isoformat(),'p_home=',round(prob,4),'injury_reports=',count)

def report(c):
    df=pd.read_sql_query('SELECT p.*,g.home_pts,g.away_pts FROM predictions p JOIN games g USING(game_id) WHERE g.home_pts IS NOT NULL AND g.away_pts IS NOT NULL',c)
    if df.empty:print('No settled predictions yet');return
    df['target']=(df.home_pts>df.away_pts).astype(int)
    for slot,g in df.groupby('slot'):
        print(slot,'n=',len(g),'brier=',round(brier_score_loss(g.target,g.prob_home),5),'logloss=',round(log_loss(g.target,g.prob_home,labels=[0,1]),5))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--db',default='nba.sqlite');s=p.add_subparsers(dest='cmd',required=True)
    for cmd in ['import-games','import-injuries']:s.add_parser(cmd).add_argument('csv')
    s.add_parser('backtest').add_argument('--first-test',default='2022-23')
    sp=s.add_parser('predict');sp.add_argument('--slot',choices=['morning','tminus60'],required=True);sp.add_argument('--day',required=True)
    s.add_parser('report');a=p.parse_args();c=connect(a.db)
    if a.cmd=='import-games':import_games(c,a.csv)
    elif a.cmd=='import-injuries':import_injuries(c,a.csv)
    elif a.cmd=='backtest':train_eval(c,a.first_test)
    elif a.cmd=='predict':snapshot(c,a.slot,a.day)
    elif a.cmd=='report':report(c)
