"""Screening-record storage: Supabase (Postgres via its REST API) when SUPABASE_URL + SUPABASE_SERVICE_KEY are set, else a local SQLite file.
Only the screening outcome and the entered patient details are stored, never the uploaded scan. Stdlib only."""
import json, os, sqlite3, urllib.error, urllib.parse, urllib.request
from datetime import datetime, timezone

FIELDS = ("report_id", "created_at", "patient_name", "patient_id", "age", "sex", "flagged", "severity")
SB_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SB_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
DB_PATH = os.environ.get("RECORDS_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "records.db"))
BACKEND = "supabase" if (SB_URL and SB_KEY) else "sqlite"


class StoreError(Exception):
    pass


# ---------------- Supabase (PostgREST) ----------------
def _sb(method, query="", body=None, prefer=None):
    h = {"apikey": SB_KEY, "Content-Type": "application/json"}
    if SB_KEY.startswith("eyJ"):                   # legacy JWT keys also go in Authorization; new sb_secret_ keys must not
        h["Authorization"] = "Bearer " + SB_KEY
    if prefer: h["Prefer"] = prefer
    req = urllib.request.Request(f"{SB_URL}/rest/v1/screenings{query}", method=method, headers=h,
                                 data=None if body is None else json.dumps(body).encode())
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read(); return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        raise StoreError(f"database error {e.code}: {e.read().decode()[:200]}")
    except Exception as e:
        raise StoreError(f"database unreachable: {e}")


# ---------------- SQLite fallback ----------------
def _lite():
    c = sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS screenings(id INTEGER PRIMARY KEY AUTOINCREMENT, report_id TEXT, created_at TEXT, patient_name TEXT,
                 patient_id TEXT, age INTEGER, sex TEXT, flagged INTEGER, severity TEXT)""")
    c.row_factory = sqlite3.Row
    return c


def _row(r):
    d = {k: r[k] for k in FIELDS}; d["flagged"] = bool(d["flagged"]); return d


def add(rec: dict) -> dict:
    rec = dict(rec); rec.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    if BACKEND == "supabase":
        out = _sb("POST", "", rec, prefer="return=representation")
        return {k: out[0][k] for k in FIELDS}
    with _lite() as c:
        c.execute(f"INSERT INTO screenings({','.join(FIELDS)}) VALUES({','.join('?' * len(FIELDS))})", [rec.get(k) for k in FIELDS])
    return rec


def recent(limit=200) -> list:
    if BACKEND == "supabase":
        return [{k: r[k] for k in FIELDS} for r in _sb("GET", f"?select={','.join(FIELDS)}&order=created_at.desc&limit={int(limit)}")]
    with _lite() as c:
        return [_row(r) for r in c.execute("SELECT * FROM screenings ORDER BY id DESC LIMIT ?", (int(limit),))]


def clear():
    if BACKEND == "supabase":
        _sb("DELETE", "?id=not.is.null")
    else:
        with _lite() as c: c.execute("DELETE FROM screenings")
