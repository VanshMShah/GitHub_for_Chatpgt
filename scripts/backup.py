from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from datetime import datetime,timezone
from src.ats.config import settings
from src.ats.store import Store
if __name__=='__main__':
    name='ats_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'.sqlite3'
    print(Store(settings.db_path).backup(Path('data/backups')/name))
