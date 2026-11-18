"""ESPN public scoreboard importer. Unofficial endpoint; may change or rate-limit."""
import time
from datetime import date, timedelta
import pandas as pd
import requests

COLUMNS=['game_id','season','tipoff_utc','home','away','home_pts','away_pts']
URL='https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'

def fetch_season(year, progress=None):
    # Cover full season window; ESPN season type distinguishes regular season.
    start=date(year,10,1)
    end=date(year+1,7,1)
    current=start
    rows=[]; failures=[]; chunks=0
    session=requests.Session()
    session.headers.update({'User-Agent':'Mozilla/5.0 NBA-Predictor-Research/1.0','Accept':'application/json'})
    while current<end:
        last=min(current+timedelta(days=6),end-timedelta(days=1))
        window=f'{current:%Y%m%d}-{last:%Y%m%d}'
        try:
            response=session.get(URL,params={'dates':window,'limit':500},timeout=18)
            response.raise_for_status()
            payload=response.json()
            for event in payload.get('events',[]):
                if str(event.get('season',{}).get('type',''))!='2':continue
                competitions=event.get('competitions',[])
                if not competitions:continue
                comp=competitions[0]
                competitors=comp.get('competitors',[])
                home=next((t for t in competitors if t.get('homeAway')=='home'),None)
                away=next((t for t in competitors if t.get('homeAway')=='away'),None)
                if not home or not away:continue
                h=home.get('team',{}).get('abbreviation')
                a=away.get('team',{}).get('abbreviation')
                when=comp.get('date') or event.get('date')
                if not h or not a or not when:continue
                ts=pd.to_datetime(when,utc=True,errors='coerce')
                if pd.isna(ts):continue
                completed=comp.get('status',{}).get('type',{}).get('completed',False)
                hp=ap=None
                if completed:
                    try:
                        hp=int(home['score']);ap=int(away['score'])
                    except (KeyError,ValueError,TypeError):continue
                rows.append({'game_id':'espn_'+str(event['id']),'season':f'{year}-{str(year+1)[-2:]}',
                             'tipoff_utc':ts.isoformat(),'home':h,'away':a,'home_pts':hp,'away_pts':ap})
        except (requests.RequestException,ValueError,KeyError) as exc:
            failures.append(f'{window}: {type(exc).__name__}: {exc}')
        chunks+=1
        if progress:progress(chunks,40)
        current=last+timedelta(days=1)
        time.sleep(0.12)
    if not rows:raise RuntimeError('Aucun match récupéré. '+' | '.join(failures[:3]))
    frame=pd.DataFrame(rows,columns=COLUMNS).drop_duplicates('game_id').sort_values('tipoff_utc')
    # A partially downloaded season cannot silently be treated as complete.
    return frame,failures
