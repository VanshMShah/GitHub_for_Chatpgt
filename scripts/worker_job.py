from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from datetime import datetime,timezone
from src.ats.config import settings
from src.ats.store import Store
from src.ats.engine import ATSEngine
from src.ats.jobs import JobManager

def main():
    kind,identity,cases=sys.argv[1:]
    store=Store(settings.db_path)
    stamp=lambda:datetime.now(timezone.utc).isoformat()
    job=store.get_state('current_job',{})
    if kind not in JobManager.allowed or job.get('id')!=identity: raise SystemExit('Invalid job identity')
    job.update(status='RUNNING',stage=kind);store.set_state('current_job',job,stamp())
    try:
        engine=ATSEngine(store)
        if kind=='benchmark':
            from src.ats.research import evaluate
            result=evaluate('5y',252,int(cases))
            store.set_state('benchmark',result,stamp())
            Path('research').mkdir(exist_ok=True)
            Path('research/latest_verified.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        else:
            operations={'sync':engine.sync,'ingest':engine.ingest,'train':engine.train,'forecast':engine.forecast,
                'settle':engine.settle,'events':engine.ingest_events,'intraday_prepare':engine.prepare_intraday,
                'intraday_train':engine.train_intraday,'daily_neural':engine.pqrnn.train_and_evaluate}
            result=operations[kind]()
        job.update(status='SUCCEEDED',result=result,finished_at=stamp());store.set_state('current_job',job,stamp())
    except Exception as exc:
        job.update(status='FAILED',error=str(exc),finished_at=stamp());store.set_state('current_job',job,stamp())
        raise
    finally:store.close()
if __name__=='__main__':main()
