"""
Dosya-tabanlı iş koşucu (eski Streamlit panelinden taşındı — kanıtlanmış desen).

Her iş için runs/ altında iki dosya:
    {key}.log   — stdout+stderr (canlı SSE tail buradan)
    {key}.done  — bitişte çıkış kodu (varlığı = iş bitti)

Komutlar src/ içinde (cwd=SRC_DIR) mevcut Python yorumlayıcısıyla çalışır;
pipeline scriptleri kendilerini __file__'den konumlandırdığından yol sorunu yok.
"""
import os
import subprocess
import sys
from datetime import datetime

import pandas as pd

WEB_DIR  = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.dirname(WEB_DIR)
ROOT     = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT, "data")
RUNS_DIR = os.path.join(ROOT, "runs")
PRED_LOG = os.path.join(DATA_DIR, "predictions_log.csv")
PY = sys.executable

os.makedirs(RUNS_DIR, exist_ok=True)

# anahtar → (etiket, açıklama, komut)
JOBS = {
    "scrape":   ("Programı Çek", "Bugünün koşu programını ve canlı oranları çeker (Selenium).",
                 [PY, "tjk_stage5_live_program.py", "--headless"]),
    "predict":  ("Tahmin Üret", "Program için full + ablation + blend tahminlerini üretir.",
                 [PY, "tjk_stage6_predict.py"]),
    "strategy": ("Strateji Üret", "En güncel tahmin gününe bahis önerileri üretir.",
                 "STRATEGY"),
    "results":  ("Sonuç Çek + Değerlendir", "Günün sonuçlarını çekip forward-test metriklerini günceller.",
                 "RESULTS"),
    "backtest": ("Strateji Backtest", "OOF olasılıklarıyla bahis stratejisi backtest'i.",
                 [PY, "tjk_stage8_betting_strategy.py", "--backtest"]),
    "retrain":  ("Yeniden Eğit", "Veri güncelle + modelleri yeniden eğit (SAATLER sürebilir).",
                 [PY, "tjk_retrain_monitor.py", "--run"]),
}


def log_path(key):
    return os.path.join(RUNS_DIR, f"{key}.log")


def done_path(key):
    return os.path.join(RUNS_DIR, f"{key}.done")


def job_status(key):
    """→ dict(status, exit_code, last_run) — status ∈ idle|running|done|failed"""
    log, done = log_path(key), done_path(key)
    if not os.path.exists(log):
        return {"status": "idle", "exit_code": None, "last_run": None}
    last_run = datetime.fromtimestamp(os.path.getmtime(log)).strftime("%d.%m.%Y %H:%M")
    if os.path.exists(done):
        try:
            code = int(open(done).read().strip() or "0")
        except Exception:
            code = 0
        return {"status": "done" if code == 0 else "failed",
                "exit_code": code, "last_run": last_run}
    return {"status": "running", "exit_code": None, "last_run": last_run}


def any_running():
    return any(job_status(k)["status"] == "running" for k in JOBS)


def start_job(key):
    """İşi arka planda başlatır. Başka iş koşuyorsa ValueError."""
    if key not in JOBS:
        raise KeyError(key)
    if any_running():
        raise ValueError("Başka bir iş çalışıyor — bitmesini bekleyin.")
    cmd = JOBS[key][2]
    log, done = log_path(key), done_path(key)
    if os.path.exists(done):
        os.remove(done)
    # Log dosyasını hemen oluştur (status=running görünsün)
    open(log, "w").close()

    if cmd == "RESULTS":
        inner = f'"{PY}" tjk_pipeline.py --only 1 && "{PY}" tjk_stage7_reconcile.py'
    elif cmd == "STRATEGY":
        latest = _latest_prediction_date()
        if latest:
            inner = f'"{PY}" tjk_stage8_betting_strategy.py --date {latest}'
        else:
            inner = 'echo "Önce tahmin üret (predictions_log.csv yok)"'
    else:
        inner = " ".join(f'"{c}"' for c in cmd)

    wrapper = f'{inner} > "{log}" 2>&1; echo $? > "{done}"'
    subprocess.Popen(wrapper, shell=True, cwd=SRC_DIR)


def _latest_prediction_date():
    if not os.path.isfile(PRED_LOG):
        return None
    try:
        d = pd.read_csv(PRED_LOG, encoding="utf-8-sig", usecols=["Tarih"])
        dates = pd.to_datetime(d["Tarih"], format="%d.%m.%Y", errors="coerce").dropna()
        if dates.empty:
            return None
        return dates.max().strftime("%Y-%m-%d")
    except Exception:
        return None


def tail(key, n=200):
    path = log_path(key)
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        return "".join(f.readlines()[-n:])
