"""Local web app: upload a raw ROI time series (T x 360) -> real Phase 1 detector (+ exploratory severity estimate).
Run from repo root:  python3 testing_data/app/server.py   then open http://localhost:8800
Stdlib http.server only (no extra dependencies)."""
import hmac, io, json, os, secrets, sys, traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np, pandas as pd, joblib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.inference import engine as E
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import store

APP, SAMPLES = os.path.join(ROOT, "testing_data", "app"), os.path.join(ROOT, "testing_data", "samples")
FINAL = os.path.join(ROOT, "final testing time series data", "scans")
KEY = pd.read_csv(os.path.join(ROOT, "testing_data", "answer_key_and_results.csv")).fillna("")
SEV = joblib.load(os.path.join(ROOT, "outputs/experiments/severity/severity_model.joblib"))
SEV_RES = json.load(open(os.path.join(ROOT, "outputs/experiments/severity/severity_results.json")))
P1 = json.load(open(os.path.join(ROOT, "outputs/experiments/phase1/metrics.json")))
MAX_BYTES = 60 * 1024 * 1024
ACCESS_CODE = os.environ.get("APP_ACCESS_CODE", "")   # if set, every /api call needs it (X-Access-Code header)
SEX = {"", "Female", "Male", "Other"}; BANDS = {None, "low", "moderate", "high"}


def clean_record(d: dict) -> dict:
    """Validate a screening record coming from the browser; report id and timestamp are set here, not by the client."""
    s = lambda k, n: str(d.get(k) or "").strip()[:n]
    age = d.get("age"); age = int(age) if str(age or "").strip().isdigit() and 0 <= int(age) <= 120 else None
    sex = s("sex", 10); sev = d.get("severity") or None
    if sex not in SEX or sev not in BANDS or not isinstance(d.get("flagged"), bool):
        raise ValueError("Invalid record.")
    now = datetime.now(timezone.utc)
    return dict(report_id=f"NC-{now:%Y%m%d}-{secrets.token_hex(2).upper()}", created_at=now.isoformat(), patient_name=s("name", 80) or None,
                patient_id=s("pid", 40) or None, age=age, sex=sex or None, flagged=d["flagged"], severity=sev if d["flagged"] else None)


def out_record(r: dict) -> dict:
    return dict(rid=r["report_id"], when=r["created_at"], name=r["patient_name"] or "", pid=r["patient_id"] or "",
                age=r["age"] if r["age"] is not None else "", sex=r["sex"] or "", flagged=bool(r["flagged"]), severity=r["severity"])


def parse_upload(name: str, raw: bytes) -> np.ndarray:
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if ext == "npy":
        return np.load(io.BytesIO(raw), allow_pickle=False)
    if ext == "npz":
        z = np.load(io.BytesIO(raw), allow_pickle=False); return z[z.files[0]]
    df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", header=None, comment="#")
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all").dropna(axis=1, how="all")  # drops a text header row/col
    return df.to_numpy(dtype=np.float64)


def info():
    k = KEY
    return dict(
        p1=dict(auroc=P1["auroc"], auprc=P1["auprc"], precision=P1["precision"], recall=P1["recall"], f1=P1["f1"],
                accuracy=(P1["true_positives"] + P1["true_negatives"]) / P1["num_test_subjects"], n=P1["num_test_subjects"],
                flagged=P1["num_subjects_flagged_anomalous"], tau_subject=P1["tau_subject"], ci=[0.353, 0.552]),
        severity=dict(holdout=SEV_RES["holdout"], cv=SEV_RES["cv_5fold_x10"]),
        samples=dict(n=len(k), correct=int(k.detector_correct.sum()), flagged=int((k.detector_output == "flagged").sum()),
                     rows=k.to_dict(orient="records")),
    )


def analyze(name: str, raw: bytes) -> dict:
    r = E.analyze(parse_upload(name, raw), name)
    out = dict(r)
    if r["flagged"]:
        pr = SEV["model"].predict_proba(np.array(r["features"])[None])[0]
        out["severity"] = dict(band=SEV["classes"][int(pr.argmax())], probs=dict(zip(SEV["classes"], map(float, pr))))
    else:
        out["severity"] = None
    row = KEY[KEY.file == os.path.basename(name)]
    if len(row):
        out["answer_key"] = row.iloc[0].to_dict()
    for k in ("features",): out.pop(k, None)
    return out


def screen(name: str, raw: bytes) -> dict:
    """Clinic-facing result: only the screening outcome and, if flagged, the severity level (no technical detail)."""
    r = analyze(name, raw)
    return dict(flagged=bool(r["flagged"]), severity=(r["severity"]["band"] if r["severity"] else None), explain=r["explain"])


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else json.dumps(body, default=float).encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)

    def _authed(self):
        if not ACCESS_CODE: return True
        if hmac.compare_digest(self.headers.get("X-Access-Code", ""), ACCESS_CODE): return True
        self._send(401, {"error": "Access code required."}); return False

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/brain.json", "/brain3d.js"):   # static assets for the 3D connection view
            return self._send(200, open(os.path.join(APP, p[1:]), "rb").read(), "application/json" if p.endswith("json") else "text/javascript")
        if p == "/healthz":
            return self._send(200, {"ok": True})
        if p == "/api/config":
            return self._send(200, {"auth": bool(ACCESS_CODE), "store": store.BACKEND})
        if p.startswith("/api/") and not self._authed(): return
        if p == "/api/comparison":
            f = os.path.join(ROOT, "final testing time series data", "comparison_results.json")
            return self._send(200, open(f, "rb").read()) if os.path.isfile(f) else self._send(404, {"error": "not found"})
        if p == "/api/records":
            try: return self._send(200, [out_record(r) for r in store.recent()])
            except store.StoreError as e:
                print('records:', e, file=sys.stderr); return self._send(503, {"error": "Records are temporarily unavailable."})
        if p in ("/", "/index.html"):
            return self._send(200, open(os.path.join(APP, "index.html"), "rb").read(), "text/html; charset=utf-8")
        if p == "/research":   # the earlier technical page, not linked from the clinic UI
            return self._send(200, open(os.path.join(APP, "research.html"), "rb").read(), "text/html; charset=utf-8")
        if p.startswith("/demo-scans/"):
            f = os.path.join(FINAL, os.path.basename(p))
            if os.path.isfile(f):
                return self._send(200, open(f, "rb").read(), "text/csv")
        if p == "/api/info":
            return self._send(200, info())
        if p.startswith("/samples/"):
            f = os.path.join(SAMPLES, os.path.basename(p))
            if os.path.isfile(f):
                return self._send(200, open(f, "rb").read(), "text/csv")
        self._send(404, {"error": "not found"})

    def do_DELETE(self):
        if self.path != "/api/records": return self._send(404, {"error": "not found"})
        if not self._authed(): return
        try: store.clear(); self._send(200, {"ok": True})
        except store.StoreError as e: print('records:', e, file=sys.stderr); self._send(503, {"error": "Records are temporarily unavailable."})

    def do_POST(self):
        if self.path == "/api/records":
            if not self._authed(): return
            try:
                n = int(self.headers.get("Content-Length", 0))
                if not 0 < n < 10_000: raise ValueError("Invalid record.")
                return self._send(200, out_record(store.add(clean_record(json.loads(self.rfile.read(n))))))
            except (ValueError, json.JSONDecodeError) as e: return self._send(400, {"error": str(e)})
            except store.StoreError as e: print('records:', e, file=sys.stderr); return self._send(503, {"error": "Records are temporarily unavailable."})
        if self.path not in ("/api/analyze", "/api/screen"):
            return self._send(404, {"error": "not found"})
        if not self._authed(): return
        fn = screen if self.path == "/api/screen" else analyze
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0 or n > MAX_BYTES:
            return self._send(400, {"error": "Empty file or file larger than 60 MB."})
        try:
            self._send(200, fn(self.headers.get("X-Filename", "upload"), self.rfile.read(n)))
        except ValueError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:  # unexpected: report, do not crash the server
            traceback.print_exc(); self._send(500, {"error": f"Could not analyse this file ({type(e).__name__}): {e}"})

    def log_message(self, *a): pass


if __name__ == "__main__":
    E.get_models()
    host, port = os.environ.get("HOST", "127.0.0.1"), int(os.environ.get("PORT", "8800"))
    print(f"Open http://localhost:{port}  (Ctrl+C to stop) | records: {store.BACKEND} | access code: {'on' if ACCESS_CODE else 'off'}")
    ThreadingHTTPServer((host, port), H).serve_forever()
