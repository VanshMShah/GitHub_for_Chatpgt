from __future__ import annotations
import asyncio
import csv
import io
import os
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from src.ats.config import settings, ROOT
from src.ats.store import Store
from src.ats.engine import ATSEngine
from src.ats.agent import BackgroundAgent
from src.ats.jobs import JobManager
from src.ats.paper.account import PaperAccount
from src.ats.live import UpstoxLiveProvider
from src.ats.paper.live import LivePaperRunner, LivePaperConfig

store=Store(settings.db_path)
engine=ATSEngine(store)
agent=BackgroundAgent(engine)
jobs=JobManager(store)
task=None
paper_process=None
paper_log=None
stamp=lambda:datetime.now(timezone.utc).isoformat()

@asynccontextmanager
async def lifespan(app):
    global task
    store.set_state('agent:running',False,stamp())
    if settings.auto_start_agent:
        task=asyncio.create_task(agent.run());store.set_state('agent:running',True,stamp())
    yield
    agent.stop()
    if task:
        task.cancel()
        try:await task
        except asyncio.CancelledError:pass
    jobs.stop()
    stop_paper_process()
    store.set_state('agent:running',False,stamp())

app=FastAPI(title='ATS Research & Paper Dashboard',version='3.0.0',lifespan=lifespan)
app.mount('/assets',StaticFiles(directory=ROOT/'dashboard'),name='assets')

@app.middleware('http')
async def access(request:Request,call_next):
    # Local by default; prevent cross-origin browser requests from changing settings.
    host=request.url.hostname
    if host not in ('127.0.0.1','localhost','testserver'):
        return JSONResponse({'detail':'Local dashboard only. Use a secured reverse proxy for remote deployment.'},status_code=403)
    if request.method not in ('GET','HEAD','OPTIONS'):
        origin=request.headers.get('origin')
        if origin and (urlsplit(origin).netloc!=request.url.netloc or urlsplit(origin).scheme!=request.url.scheme):
            return JSONResponse({'detail':'Cross-origin control requests are rejected'},status_code=403)
    if request.url.path.startswith('/api/') and settings.api_token:
        supplied=request.headers.get('authorization','')
        if not secrets.compare_digest(supplied,'Bearer '+settings.api_token):
            return JSONResponse({'detail':'Enter the local API token in Settings'},status_code=401)
    response=await call_next(request)
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Referrer-Policy']='same-origin'
    return response

@app.get('/',response_class=HTMLResponse)
async def home():return (ROOT/'dashboard'/'index.html').read_text(encoding='utf-8')
@app.get('/favicon.ico')
async def favicon():return Response(status_code=204)

@app.get('/api/status')
async def status():
    result=await asyncio.to_thread(engine.status)
    result.update(job=jobs.status(),worker={'running':bool(task and not task.done()),'last_success':store.get_state('agent:last_success'),'last_failure':store.get_state('agent:last_failure')},paper=store.get_state('live_paper_heartbeat',{}))
    return result
@app.get('/api/health')
async def health():
    return {'ok':True,'market_hours':engine.is_market_hours(),'live_configured':engine.live_provider.configured,'job':jobs.status(),'calendar_years':[2026],'orders':'paper only'}
@app.get('/api/history')
async def history():return engine.history(120)
@app.get('/api/forecasts')
async def forecasts():return engine.ledger()
@app.get('/api/performance')
async def performance():return {**engine.performance(),'benchmark':store.get_state('benchmark')}
@app.get('/api/models')
async def models():return engine.model_status()
@app.get('/api/pqrnn')
async def pqrnn_status():return engine.pqrnn.status()
@app.get('/api/diagnostics')
async def diagnostics():return [dict(r) for r in store.query('SELECT * FROM diagnostics ORDER BY id DESC LIMIT 100')]
@app.get('/api/policy/history')
async def policy_history():return [dict(r) for r in store.query('SELECT * FROM policy_history ORDER BY id DESC LIMIT 100')]
@app.get('/api/jobs/current')
async def current_job():return jobs.status()

class JobInput(BaseModel):
    kind:str
    cases:int=Field(default=30,ge=5,le=300)
async def start_job(kind,cases=30):
    if task and not task.done():raise HTTPException(409,'Stop the background worker before running a manual job')
    if paper_process and paper_process.poll() is None:raise HTTPException(409,'Stop paper runner before changing models or data')
    try:return jobs.start(kind,cases)
    except RuntimeError as exc:raise HTTPException(409,str(exc))
    except ValueError as exc:raise HTTPException(400,str(exc))
@app.post('/api/jobs',status_code=202)
async def job(body:JobInput):return await start_job(body.kind,body.cases)
@app.post('/api/jobs/stop')
async def stop_job():return jobs.stop()
# Original routes retained; these now return a job ID immediately.
@app.post('/api/ingest',status_code=202)
async def ingest():return await start_job('ingest')
@app.post('/api/train',status_code=202)
async def train():return await start_job('train')
@app.post('/api/forecast',status_code=202)
async def forecast():return await start_job('forecast')
@app.post('/api/sync',status_code=202)
async def sync():return await start_job('sync')
@app.post('/api/settle',status_code=202)
async def settle():return await start_job('settle')
@app.post('/api/pqrnn/train',status_code=202)
async def train_neural():return await start_job('daily_neural')
@app.post('/api/live')
async def live():return await asyncio.to_thread(engine.ingest_live)

class RiskUpdate(BaseModel):level:str
@app.post('/api/risk')
async def risk(body:RiskUpdate):
    try:return {'risk':engine.risk.set_level(body.level)}
    except ValueError as exc:raise HTTPException(400,str(exc))
@app.post('/api/agent/start')
async def start_agent():
    global task
    if task and not task.done():return {'status':'already_running'}
    if jobs.status().get('status') in ('RUNNING','QUEUED'):raise HTTPException(409,'Wait for the manual job to finish')
    if paper_process and paper_process.poll() is None:raise HTTPException(409,'Stop paper runner first')
    agent.stop_event.clear();task=asyncio.create_task(agent.run());store.set_state('agent:running',True,stamp())
    return {'status':'running'}
@app.post('/api/agent/stop')
async def stop_agent():
    global task
    agent.stop()
    if task and not task.done():
        task.cancel()
        try:await task
        except asyncio.CancelledError:pass
    task=None;store.set_state('agent:running',False,stamp());return {'status':'stopped'}

class Preferences(BaseModel):
    risk_level:str='medium'
    default_view:str='overview'
@app.get('/api/settings')
async def configuration():
    return {'instrument':settings.execution_key,'reference_instrument':settings.upstox_nifty_key,'token_present':bool(settings.upstox_access_token),
        'api_auth_enabled':bool(settings.api_token),'paper_enabled':settings.live_paper_enabled,'daily_horizon':'1D','intraday_interval':60,
        'db_path':str(settings.db_path),'calendar_years':[2026],'restart_required_for_env':True,
        'fee_bps':float(os.getenv('ATS_PAPER_FEE_BPS','0')),'fixed_fee':float(os.getenv('ATS_PAPER_FIXED_FEE','0')),
        'lot_size':int(os.getenv('ATS_PAPER_LOT_SIZE','1')),'allow_short':os.getenv('ATS_PAPER_ALLOW_SHORT','false').lower()=='true',
        'paper_policy':'flatten five minutes before close; failed exits retained and flagged; kill switch halts all fills',
        'preferences':store.get_state('preferences',{})}
@app.post('/api/settings')
async def preferences(body:Preferences):
    try:engine.risk.set_level(body.risk_level)
    except ValueError as exc:raise HTTPException(400,str(exc))
    if body.default_view not in ('overview','history','performance','models','paper','health','settings'):raise HTTPException(400,'Invalid view')
    store.set_state('preferences',body.model_dump(),stamp());return body.model_dump()

def paper_account():return PaperAccount(store,float(os.getenv('ATS_PAPER_FEE_BPS','0')),float(os.getenv('ATS_PAPER_FIXED_FEE','0')))
@app.get('/api/paper')
async def paper():
    return {**paper_account().snapshot(),'heartbeat':store.get_state('live_paper_heartbeat',{}),'running':bool(paper_process and paper_process.poll() is None),'halted':Path(os.getenv('ATS_PAPER_KILL_SWITCH','data/PAPER_KILL')).exists()}

def stop_paper_process():
    global paper_process,paper_log
    if paper_process and paper_process.poll() is None:
        paper_process.terminate()
        try:paper_process.wait(timeout=5)
        except __import__('subprocess').TimeoutExpired:paper_process.kill();paper_process.wait()
    paper_process=None
    if paper_log:paper_log.close();paper_log=None
@app.post('/api/paper/start')
async def start_paper():
    global paper_process,paper_log
    if task and not task.done():raise HTTPException(409,'Stop daily worker first')
    if jobs.status().get('status') in ('RUNNING','QUEUED'):raise HTTPException(409,'Wait for the job to finish')
    if paper_process and paper_process.poll() is None:return {'status':'already_running'}
    if not settings.live_paper_enabled or not settings.execution_key or not settings.upstox_access_token:
        raise HTTPException(400,'Configure paper enable, execution instrument and Upstox access token in .env; restart')
    meta=store.get_state('pqrnn_intraday_meta',{})
    if not meta.get('promoted') or meta.get('instrument_key')!=settings.execution_key or not engine.pqrnn_intraday.path.exists():
        raise HTTPException(400,'Prepare one-minute data and pass intraday model validation for this instrument first')
    import subprocess,sys
    (ROOT/'logs').mkdir(exist_ok=True)
    paper_log=(ROOT/'logs'/'paper_runner.log').open('a',encoding='utf-8')
    env=os.environ.copy();env['ATS_DB_PATH']=str(store.path.resolve())
    paper_process=subprocess.Popen([sys.executable,str(ROOT/'scripts'/'run_live_paper.py')],cwd=ROOT,env=env,stdout=paper_log,stderr=subprocess.STDOUT)
    return {'status':'running','mode':'PAPER_ONLY'}
@app.post('/api/paper/stop')
async def stop_paper():
    stop_paper_process();return {'status':'stopped','position_policy':'positions retained; no implicit fill'}
class KillInput(BaseModel):halt:bool
@app.post('/api/paper/kill')
async def kill(body:KillInput):
    path=Path(os.getenv('ATS_PAPER_KILL_SWITCH','data/PAPER_KILL'))
    if body.halt:path.parent.mkdir(parents=True,exist_ok=True);path.touch()
    else:path.unlink(missing_ok=True)
    return {'halted':path.exists()}
@app.post('/api/paper/flatten')
async def flatten():
    if paper_process and paper_process.poll() is None:raise HTTPException(409,'Stop paper runner before a manual exit')
    if not settings.live_paper_enabled:raise HTTPException(400,'Paper mode disabled')
    cfg=LivePaperConfig(enable=True,execution_instrument_key=settings.execution_key,lot_size=int(os.getenv('ATS_PAPER_LOT_SIZE','1')),
        lot_size_confirmed=bool(os.getenv('ATS_PAPER_LOT_SIZE')),fee_bps=float(os.getenv('ATS_PAPER_FEE_BPS','0')),fixed_fee=float(os.getenv('ATS_PAPER_FIXED_FEE','0')),
        kill_switch_path=os.getenv('ATS_PAPER_KILL_SWITCH','data/PAPER_KILL'))
    runner=LivePaperRunner(engine,UpstoxLiveProvider(settings.upstox_access_token,settings.execution_key),cfg)
    try:
        def perform():
            with engine._process_lock:return runner.flatten()
        return await asyncio.to_thread(perform)
    except Exception as exc:raise HTTPException(400,str(exc))

@app.get('/api/export/{kind}')
async def export(kind:str):
    if kind=='forecasts':rows=engine.ledger(10000)
    elif kind=='orders':rows=paper_account().snapshot()['orders']
    elif kind=='fills':rows=paper_account().snapshot()['fills']
    else:raise HTTPException(404,'Unknown export')
    output=io.StringIO()
    if rows:
        writer=csv.DictWriter(output,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return Response(output.getvalue(),media_type='text/csv',headers={'Content-Disposition':f'attachment; filename="ats_{kind}.csv"'})
@app.post('/api/backup')
async def backup():
    name='ats_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'.sqlite3'
    path=store.backup(ROOT/'data'/'backups'/name)
    return {'saved':path,'note':'SQLite backup only. Preserve models and .env separately; protect credentials.'}
