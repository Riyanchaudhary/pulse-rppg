"""Small SQLite metadata index; large signal arrays stay in per-analysis JSON files."""
import sqlite3
from contextlib import closing
from pathlib import Path
from .api_config import DB_PATH

def connect():
    DB_PATH.parent.mkdir(parents=True,exist_ok=True)
    c=sqlite3.connect(DB_PATH); c.row_factory=sqlite3.Row
    c.execute('''CREATE TABLE IF NOT EXISTS analyses (id TEXT PRIMARY KEY, filename TEXT NOT NULL, created_at TEXT, duration REAL, width INTEGER, height INTEGER, fps REAL, frame_count INTEGER, status TEXT NOT NULL, interpretation TEXT, selected_roi TEXT, result_path TEXT)''')
    c.commit(); return c

def save(data,path):
    m=data['metadata']
    with closing(connect()) as c:
        c.execute('''INSERT OR REPLACE INTO analyses VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',(data['id'],m['filename'],data.get('created_at'),m['duration'],m['width'],m['height'],m['fps'],m['total_frames'],'complete',data.get('interpretation'),data.get('selected_roi'),str(path)))
        c.commit()

def list_history():
    with closing(connect()) as c: return [dict(r) for r in c.execute('SELECT id,filename,created_at,duration,selected_roi,interpretation FROM analyses WHERE status="complete" ORDER BY created_at DESC')]
