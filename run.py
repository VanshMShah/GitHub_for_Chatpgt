from pathlib import Path
import argparse
import os
if __name__=='__main__':
    os.chdir(Path(__file__).resolve().parent)
    parser=argparse.ArgumentParser(description='Local ATS dashboard')
    parser.add_argument('--demo',action='store_true',help='Offline synthetic demo using separate state/model files')
    args=parser.parse_args()
    if args.demo:
        os.environ.update(ATS_DB_PATH='data/demo_state.sqlite3',ATS_MODEL_DIR='models/demo',ATS_AUTOSTART='false',ATS_UPSTOX_ACCESS_TOKEN='',ATS_UPSTOX_EXECUTION_KEY='',ATS_LIVE_PAPER_ENABLE='false')
        from src.ats.demo import seed_demo
        seed_demo()
    import uvicorn
    uvicorn.run('src.ats.api:app',host='127.0.0.1',port=8000,reload=False)
