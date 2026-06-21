#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  TJK CANLI ZAMANLAYICI — VPS otomasyon daemon'u
================================================================================
  Günlük forward-test döngüsünü TAM OTOMATİK yürütür (mevcut scriptleri subprocess
  ile çağırır; yeni iş mantığı yok). Europe/Istanbul saatiyle çalışır.

  Akış:
    1) SABAH        : Stage 5 (program çek) → Stage 6 (tahmin)
                      → ablation + full tahminleri hemen hazır.
    2) YARIŞ-BAŞINA : her koşunun post saatinden (LEAD) dk önce yeniden
                      Stage 5 (taze oran) + Stage 6 (yeniden tahmin).
                      Yakın tetikler tek dalgada birleştirilir; geçmişler atlanır.
    3) AKŞAM        : son yarıştan (+tampon) sonra sonuçları çek (pipeline --only 1)
                      + Stage 7 (reconcile) → forward-test metrikleri güncellenir.

  ⚠️  GATE: "post − LEAD" oran tazeliği YALNIZCA program sayfası canlı (hareketli)
      oran veriyorsa fayda sağlar. Sayfa hep "muhtemel ganyan" veriyorsa geç çekim
      oran açısından fark etmez — bu durumda ablation (oran-bağımsız) model resmî
      metrik olarak kalır. Bir yarış gününde doğrulayın (bkz. --dry-run + iki çekim).

  Kullanım:
    python tjk_live_scheduler.py --date 2026-06-21 --lead 10 --headless
    python tjk_live_scheduler.py --dry-run           # sadece planı yazdır, bekleme
================================================================================
"""
import os
import sys
import time
import argparse
import subprocess
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo

import pandas as pd

IST      = ZoneInfo("Europe/Istanbul")
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))     # scriptler burada (cwd)
ROOT     = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT, "data")
RUNS_DIR = os.path.join(ROOT, "runs")
PROGRAM_CSV = os.path.join(DATA_DIR, "program_tablo.csv")
LOG_FILE = os.path.join(RUNS_DIR, "scheduler.log")
PY = sys.executable

os.makedirs(RUNS_DIR, exist_ok=True)


def log(msg):
    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd, desc, headless=False):
    """Bir scripti src/ içinde çalıştır; hatayı yut, logla (daemon ölmez)."""
    env = dict(os.environ)
    if headless:
        env["TJK_HEADLESS"] = "1"
    log(f"▶ {desc}: {' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, cwd=SRC_DIR, env=env)
        log(f"  ↳ bitiş kodu {r.returncode}")
        return r.returncode == 0
    except Exception as e:
        log(f"  ✗ hata: {e}")
        return False


def scrape_and_predict(date_ddmmyyyy, headless):
    ok = run([PY, "tjk_stage5_live_program.py", "--date", date_ddmmyyyy]
             + (["--headless"] if headless else []), "Program çek (Stage 5)", headless)
    if ok:
        run([PY, "tjk_stage6_predict.py"], "Tahmin üret (Stage 6)", headless)


def read_post_times(date_disp):
    """program_tablo.csv'den o günün koşu post saatlerini (Istanbul datetime) döndürür."""
    if not os.path.isfile(PROGRAM_CSV):
        return []
    df = pd.read_csv(PROGRAM_CSV, encoding="utf-8-sig")
    if "Kosu_Saati" not in df.columns:
        return []
    df = df[df["Tarih"].astype(str) == date_disp]
    saatler = sorted(df["Kosu_Saati"].dropna().astype(str).unique())
    day = datetime.strptime(date_disp, "%d.%m.%Y").date()
    posts = []
    for s in saatler:
        try:
            hh, mm = s.split(":")
            posts.append(datetime.combine(day, dtime(int(hh), int(mm)), tzinfo=IST))
        except Exception:
            continue
    return posts


def build_triggers(posts, lead_min, merge_min=5):
    """post − lead tetikleri; birbirine ≤merge_min dk olanları tek dalgada birleştir."""
    raw = sorted(p - timedelta(minutes=lead_min) for p in posts)
    merged = []
    for t in raw:
        if merged and (t - merged[-1]) <= timedelta(minutes=merge_min):
            continue  # önceki dalga bu yarışı da kapsıyor
        merged.append(t)
    return merged


def main():
    ap = argparse.ArgumentParser(description="TJK canlı zamanlayıcı (VPS daemon)")
    ap.add_argument("--date", help="YYYY-MM-DD (varsayılan: bugün, Istanbul)")
    ap.add_argument("--lead", type=int, default=10, help="Post'tan kaç dk önce oran çek")
    ap.add_argument("--results-buffer", type=int, default=30,
                    help="Son yarıştan kaç dk sonra sonuç çek + reconcile")
    ap.add_argument("--headless", action="store_true", help="Selenium headless (TJK_HEADLESS=1)")
    ap.add_argument("--no-results", action="store_true", help="Akşam sonuç/reconcile adımını atla")
    ap.add_argument("--dry-run", action="store_true",
                    help="Sadece planı yazdır (çekme/bekleme yok) — gate doğrulaması için")
    args = ap.parse_args()

    today = datetime.now(IST).date()
    date_obj = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else today
    date_ddmmyyyy = date_obj.strftime("%d/%m/%Y")
    date_disp = date_obj.strftime("%d.%m.%Y")

    log("█" * 60)
    log(f"CANLI ZAMANLAYICI başladı — tarih={date_disp} lead={args.lead}dk "
        f"headless={args.headless} dry_run={args.dry_run}")

    # ── DRY-RUN: yalnız mevcut program_tablo'dan planı göster ──
    if args.dry_run:
        posts = read_post_times(date_disp)
        if not posts:
            log("  (program_tablo.csv'de bu güne ait Kosu_Saati yok — önce Stage 5 çalıştır)")
            return
        trig = build_triggers(posts, args.lead)
        now = datetime.now(IST)
        log(f"  {len(posts)} koşu post saati: " +
            ", ".join(p.strftime('%H:%M') for p in posts))
        log(f"  {len(trig)} oran-çekim tetiği (post−{args.lead}dk):")
        for t in trig:
            durum = "GEÇMİŞ (atlanır)" if t <= now else f"{(t-now)}"
            log(f"    {t.strftime('%H:%M')}  →  {durum}")
        if not args.no_results:
            res = max(posts) + timedelta(minutes=args.results_buffer)
            log(f"  Sonuç+reconcile tetiği: {res.strftime('%H:%M')}")
        return

    # ── 1) SABAH: ilk program + tahmin ──
    scrape_and_predict(date_ddmmyyyy, args.headless)

    # ── 2) YARIŞ-BAŞINA tetikler ──
    posts = read_post_times(date_disp)
    if not posts:
        log("  ⚠ Post saati bulunamadı; yalnız sabah çekimiyle yetinildi.")
        return
    triggers = build_triggers(posts, args.lead)
    log(f"  {len(posts)} koşu, {len(triggers)} oran-çekim dalgası planlandı.")

    for t in triggers:
        now = datetime.now(IST)
        wait = (t - now).total_seconds()
        if wait <= 0:
            log(f"  ⏭ {t.strftime('%H:%M')} geçmiş — atlanıyor.")
            continue
        log(f"  ⏳ {t.strftime('%H:%M')} bekleniyor ({int(wait)}s)...")
        time.sleep(wait)
        scrape_and_predict(date_ddmmyyyy, args.headless)

    # ── 3) AKŞAM: sonuç + reconcile ──
    if not args.no_results:
        res_at = max(posts) + timedelta(minutes=args.results_buffer)
        wait = (res_at - datetime.now(IST)).total_seconds()
        if wait > 0:
            log(f"  ⏳ Sonuçlar için {res_at.strftime('%H:%M')} bekleniyor ({int(wait)}s)...")
            time.sleep(wait)
        run([PY, "tjk_pipeline.py", "--only", "1"], "Sonuç çek (Stage 1)", args.headless)
        run([PY, "tjk_stage7_reconcile.py"], "Reconcile (Stage 7)", args.headless)

    log("CANLI ZAMANLAYICI tamamlandı.")
    log("█" * 60)


if __name__ == "__main__":
    main()
