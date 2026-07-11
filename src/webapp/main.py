"""
TJK Tahmin Paneli — FastAPI backend.

Aynı uygulama hem kökte hem /ganyan altında yanıt verir (nginx'in trailing-slash
davranışından bağımsız). Streamlit'in yerini alır: canlı SSE log akışı, yarış
detayına inen kartlar, mobil uyumlu tek-sayfa arayüz.

Çalıştırma (lokal):  uvicorn webapp.main:app --port 8501  (cwd=src/)
Docker CMD:          uvicorn src.webapp.main:app --host 0.0.0.0 --port 8501
"""
import asyncio
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import data, jobs

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT    = data.ROOT
REPORTS = data.REPORTS

core = FastAPI(title="TJK Tahmin Paneli", docs_url=None, redoc_url=None)
core.mount("/static", StaticFiles(directory=os.path.join(WEB_DIR, "static")), name="static")
if os.path.isdir(REPORTS):
    core.mount("/reports", StaticFiles(directory=REPORTS), name="reports")


@core.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "templates", "index.html"))


@core.get("/health")
def health():
    return {"ok": True}


# ── Veri API'leri ────────────────────────────────────────────────────────────
@core.get("/api/predictions")
def api_predictions(date: str | None = None):
    return data.predictions_payload(date)


@core.get("/api/performance")
def api_performance():
    return data.performance_payload()


@core.get("/api/strategy")
def api_strategy(bet_type: str | None = None):
    return data.strategy_payload(bet_type)


@core.get("/api/models")
def api_models():
    return data.models_payload()


# ── İş API'leri ──────────────────────────────────────────────────────────────
@core.get("/api/jobs")
def api_jobs():
    return {key: {"label": label, "desc": desc, **jobs.job_status(key)}
            for key, (label, desc, _) in jobs.JOBS.items()}


@core.post("/api/jobs/{key}/start")
def api_job_start(key: str):
    if key not in jobs.JOBS:
        raise HTTPException(404, "Bilinmeyen iş")
    try:
        jobs.start_job(key)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"ok": True}


@core.get("/api/jobs/{key}/log")
def api_job_log(key: str):
    if key not in jobs.JOBS:
        raise HTTPException(404, "Bilinmeyen iş")
    return {"log": jobs.tail(key), **jobs.job_status(key)}


@core.get("/api/jobs/{key}/stream")
async def api_job_stream(key: str):
    """Canlı log akışı (SSE). .done görünene kadar log dosyasını tail eder."""
    if key not in jobs.JOBS:
        raise HTTPException(404, "Bilinmeyen iş")

    async def gen():
        log, done = jobs.log_path(key), jobs.done_path(key)
        pos = 0
        idle_ticks = 0
        while True:
            chunk = ""
            if os.path.exists(log):
                with open(log, encoding="utf-8", errors="replace") as f:
                    f.seek(pos)
                    chunk = f.read()
                    pos = f.tell()
            if chunk:
                idle_ticks = 0
                for line in chunk.splitlines():
                    yield f"data: {line}\n\n"
            else:
                idle_ticks += 1
                if idle_ticks % 15 == 0:
                    yield ": heartbeat\n\n"  # proxy bağlantısı kopmasın
            if os.path.exists(done) and not chunk:
                code = open(done).read().strip() or "0"
                yield f"event: done\ndata: {code}\n\n"
                return
            await asyncio.sleep(1)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ── Dış uygulama: hem / hem /ganyan altında aynı içerik ─────────────────────
app = FastAPI(docs_url=None, redoc_url=None)
app.mount("/ganyan", core)
app.mount("/", core)
