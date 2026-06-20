#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  TJK — YENİDEN EĞİTİM + İZLEME (Continual Learning Orkestratörü)
================================================================================
  "Öğrenme" burada veri büyüdükçe periyodik YENİDEN EĞİTİM + canlı performans
  İZLEME demektir (MLOps). LLM "sezgisi" YOKTUR — tahmin daima eğitilmiş
  modellerden gelir.

  Kararlar:
    • Kadans: son yeniden eğitimden bu yana RETRAIN_EVERY_DAYS gün geçtiyse,
    • Drift : canlı kümülatif P@1 (full, winner) MIN_P1_FLOOR altına düştüyse
    → yeniden eğitim "gerekli" sayılır.

  Kullanım:
      python tjk_retrain_monitor.py --status        # sadece durum (read-only)
      python tjk_retrain_monitor.py --update-data    # Stage 1→2→3 (Selenium, uzun)
      python tjk_retrain_monitor.py --retrain        # Stage 4 (full + ablation)
      python tjk_retrain_monitor.py --run            # gerekliyse: update + retrain

  Zamanlama (cron örneği — her gün 09:00, tahmin; Pazartesi 03:00, retrain):
      0 9 * * *  cd /path/Ganyan && .venv/bin/python tjk_stage5_live_program.py --headless && .venv/bin/python tjk_stage6_predict.py
      0 3 * * 1  cd /path/Ganyan && .venv/bin/python tjk_retrain_monitor.py --run
================================================================================
"""

import os
import sys
import json
import subprocess
from datetime import datetime, timedelta
import pandas as pd

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
STATE_FILE  = os.path.join(BASE_DIR, "retrain_state.json")
LOG_FILE    = os.path.join(BASE_DIR, "retrain_log.csv")
PERF_CSV    = os.path.join(BASE_DIR, "live_performance.csv")
PY          = sys.executable

# ── Politika ──
RETRAIN_EVERY_DAYS = 7      # sabit kadans
MIN_P1_FLOOR       = 0.30   # canlı P@1 (full, winner) bu eşiğin altına düşerse drift


def _load_state():
    if os.path.isfile(STATE_FILE):
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"last_retrain": None}


def _save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _days_since_last(state):
    if not state.get("last_retrain"):
        return None
    last = datetime.fromisoformat(state["last_retrain"])
    return (datetime.now() - last).days


def _live_p1_full():
    """Canlı kümülatif P@1 (full, winner) — yoksa None."""
    if not os.path.isfile(PERF_CSV):
        return None
    perf = pd.read_csv(PERF_CSV, encoding="utf-8-sig")
    sub = perf[(perf["scope"] == "kümülatif") & (perf["variant"] == "full")]
    if sub.empty or "P@1_winner" not in sub.columns:
        return None
    return float(sub["P@1_winner"].iloc[0])


def is_retrain_due(state, verbose=True):
    days = _days_since_last(state)
    p1 = _live_p1_full()
    reasons = []
    if days is None:
        reasons.append("hiç yeniden eğitim kaydı yok")
    elif days >= RETRAIN_EVERY_DAYS:
        reasons.append(f"son eğitimden {days} gün geçti (≥{RETRAIN_EVERY_DAYS})")
    if p1 is not None and p1 < MIN_P1_FLOOR:
        reasons.append(f"canlı P@1={p1:.1%} < eşik {MIN_P1_FLOOR:.0%} (drift)")
    due = len(reasons) > 0
    if verbose:
        print(f"  Son eğitimden bu yana: {days if days is not None else '—'} gün")
        print(f"  Canlı P@1 (full):      {f'{p1:.1%}' if p1 is not None else '—'}")
        print(f"  Yeniden eğitim gerekli mi? {'EVET → ' + '; '.join(reasons) if due else 'HAYIR'}")
    return due, reasons


def _run(cmd, desc):
    print(f"\n  ▶ {desc}: {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=BASE_DIR)
    if r.returncode != 0:
        raise SystemExit(f"  ❌ Başarısız ({desc}), kod={r.returncode}")
    print(f"  ✓ Tamam: {desc}")


def update_data():
    """Yeni sonuçları çek + feature matrix'i tazele (Stage 1→2→3)."""
    _run([PY, "tjk_pipeline.py", "--from", "1"], "Veri güncelle (Stage 1→3)")


def retrain():
    """Modelleri yeniden eğit: tam + ablation."""
    _run([PY, "tjk_stage4_modeling.py"], "Yeniden eğitim (tam)")
    _run([PY, "tjk_stage4_modeling.py", "--ablation"], "Yeniden eğitim (ablation)")
    state = _load_state()
    state["last_retrain"] = datetime.now().isoformat()
    _save_state(state)
    # Log
    p1 = _live_p1_full()
    row = pd.DataFrame([{"timestamp": state["last_retrain"], "live_P1_full": p1}])
    if os.path.isfile(LOG_FILE):
        row = pd.concat([pd.read_csv(LOG_FILE), row], ignore_index=True)
    row.to_csv(LOG_FILE, index=False, encoding="utf-8-sig")
    print(f"\n  ✅ Yeniden eğitim tamam. Durum güncellendi → {STATE_FILE}")


def main():
    args = sys.argv[1:]
    print("█" * 60)
    print("  TJK — YENİDEN EĞİTİM + İZLEME")
    print("█" * 60)
    state = _load_state()

    if "--status" in args or not args:
        is_retrain_due(state)
        return
    if "--update-data" in args:
        update_data()
    if "--retrain" in args:
        retrain()
        return
    if "--run" in args:
        if "--update-data" not in args:
            update_data()
        due, _ = is_retrain_due(state)
        if due:
            retrain()
        else:
            print("\n  Yeniden eğitim gerekmiyor — atlandı.")


if __name__ == "__main__":
    main()
