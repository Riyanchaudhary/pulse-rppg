import os
from pathlib import Path
ROOT=Path(os.getenv("PULSE_DATA_DIR",Path(__file__).resolve().parents[1]/"data"))
DB_PATH=ROOT/"pulse.sqlite3"
