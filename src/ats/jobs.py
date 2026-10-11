"""One supervised subprocess per expensive API job; progress remains queryable."""
import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from src.ats.config import ROOT

class JobManager:
    allowed={'sync','ingest','train','forecast','settle','events','intraday_prepare','intraday_train','daily_neural','benchmark'}
    def __init__(self,store):
        self.store=store; self.process=None; self.lock=threading.Lock(); self.log=None
        old=store.get_state('current_job',{})
        if old.get('status') in ('RUNNING','QUEUED'):
            old.update(status='INTERRUPTED',error='Previous API stopped; inspect state and retry',finished_at=self.now())
            store.set_state('current_job',old,self.now())
    @staticmethod
    def now(): return datetime.now(timezone.utc).isoformat()
    def start(self,kind,cases=30):
        if kind not in self.allowed: raise ValueError('Unknown job')
        if self.store.get_state('demo_ready') and kind not in ('forecast','settle','train'):
            raise ValueError('Offline demo blocks external jobs. Restart without --demo for real data.')
        with self.lock:
            if self.process and self.process.poll() is None: raise RuntimeError('Another job is running')
            if self.log: self.log.close()
            identity=uuid.uuid4().hex
            job={'id':identity,'kind':kind,'status':'QUEUED','started_at':self.now()}
            self.store.set_state('current_job',job,self.now())
            logdir=ROOT/'logs';logdir.mkdir(exist_ok=True)
            self.log=(logdir/f'job_{identity}.log').open('w',encoding='utf-8')
            env=os.environ.copy(); env.update(ATS_DB_PATH=str(self.store.path.resolve()),ATS_AUTOSTART='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            try:
                self.process=subprocess.Popen([sys.executable,str(ROOT/'scripts'/'worker_job.py'),kind,identity,str(cases)],cwd=ROOT,env=env,stdout=self.log,stderr=subprocess.STDOUT)
            except Exception as exc:
                job.update(status='FAILED',error=str(exc));self.store.set_state('current_job',job,self.now());raise
            return job
    def status(self):
        job=self.store.get_state('current_job',{})
        if self.process and self.process.poll() is not None and job.get('status') in ('RUNNING','QUEUED'):
            job.update(status='FAILED',error=f'Worker exited with code {self.process.returncode}; see job log',finished_at=self.now())
            self.store.set_state('current_job',job,self.now())
        return job
    def stop(self):
        if self.process and self.process.poll() is None:
            # Termination abandons uncommitted work; SQLite transactions preserve committed data.
            self.process.terminate()
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
            job=self.store.get_state('current_job',{})
            job.update(status='INTERRUPTED',finished_at=self.now(),error='Stopped by user; committed records retained')
            self.store.set_state('current_job',job,self.now())
        if self.log:self.log.close();self.log=None
        return self.status()
