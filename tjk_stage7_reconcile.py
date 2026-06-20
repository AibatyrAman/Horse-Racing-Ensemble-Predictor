#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  TJK AŞAMA 7 — SONUÇ EŞLEŞTİRME + CANLI PERFORMANS (forward-test değerlendirme)
================================================================================
  Tahminleri (predictions_log.csv) gerçek sonuçlarla (yaris_ana_tablo.csv)
  eşleştirir ve her model varyantı (full/abl) için canlı P@1 / P@3 / ROI hesaplar.
  Eğitimdeki metrik fonksiyonları (precision_at_k_per_race, calculate_roi) aynen
  kullanılır → tutarlı kıyas. ROI yalnız Is_Winner için (kazanma bahsi).

  Çıktılar:
      live_performance.csv   – varyant × hedef × (günlük + kümülatif) metrik
      live_performance.md     – okunabilir özet

  Kullanım:  python tjk_stage7_reconcile.py
================================================================================
"""

import os
import numpy as np
import pandas as pd

from tjk_stage3_feature_engineering import extract_at_id_from_url
from tjk_stage4_modeling import precision_at_k_per_race, calculate_roi

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
PRED_LOG  = os.path.join(BASE_DIR, "predictions_log.csv")
YARIS_CSV = os.path.join(BASE_DIR, "yaris_ana_tablo.csv")
OUT_CSV   = os.path.join(BASE_DIR, "live_performance.csv")
OUT_MD    = os.path.join(BASE_DIR, "live_performance.md")

# (varyant etiketi, winner prob, top3 prob)
VARIANTS = [
    ("full", "prob_winner_full", "prob_top3_full"),
    ("abl",  "prob_winner_abl",  "prob_top3_abl"),
]


def load_actuals():
    """yaris_ana_tablo'dan (Unique_Race_ID, at_id) → gerçek Is_Winner/Is_Top3 + final Ganyan."""
    y = pd.read_csv(YARIS_CSV, encoding="utf-8-sig")
    y["Tarih_dt"] = pd.to_datetime(y["Tarih"], format="%d.%m.%Y", errors="coerce")
    y["Siralama"] = pd.to_numeric(y["Siralama"], errors="coerce")
    y["at_id"] = y["At_URL"].apply(extract_at_id_from_url)
    y = y.dropna(subset=["at_id"]); y["at_id"] = y["at_id"].astype(int)
    y["Unique_Race_ID"] = (
        y["Tarih_dt"].dt.strftime("%Y%m%d") + "_" +
        y["Sehir"].astype(str).str.strip() + "_" + y["Kosu_ID"].astype(str)
    )
    y["actual_Is_Winner"] = (y["Siralama"] == 1).astype(int)
    y["actual_Is_Top3"]   = (y["Siralama"] <= 3).astype(int)
    y["Ganyan_final"]     = pd.to_numeric(y["Ganyan"], errors="coerce")
    return y[["Unique_Race_ID", "at_id", "actual_Is_Winner", "actual_Is_Top3", "Ganyan_final"]]


def evaluate(df_eval, prob_w, prob_t3):
    """Bir varyant için P@1/P@3 (winner & top3) + ROI (winner) döndürür."""
    res = {}
    # Winner
    res["P@1_winner"] = precision_at_k_per_race(df_eval, prob_w, "actual_Is_Winner", k=1)
    # Top3
    res["P@1_top3"]   = precision_at_k_per_race(df_eval, prob_t3, "actual_Is_Top3", k=1)
    res["P@3_top3"]   = precision_at_k_per_race(df_eval, prob_t3, "actual_Is_Top3", k=3)
    # ROI (yalnız kazanma bahsi; final ganyan ödeme)
    roi = calculate_roi(df_eval, prob_w, "Ganyan_final", "actual_Is_Winner", strategy="top1")
    res["ROI_winner_top1"] = roi["roi"]
    res["n_bets"]          = roi["n_bets"]
    res["win_rate_top1"]   = roi["win_rate"]
    return res


def main():
    if not os.path.isfile(PRED_LOG):
        raise SystemExit(f"[HATA] {PRED_LOG} yok. Önce: python tjk_stage6_predict.py")

    pred = pd.read_csv(PRED_LOG, encoding="utf-8-sig")
    actual = load_actuals()

    merged = pred.merge(actual, on=["Unique_Race_ID", "at_id"], how="inner")
    if merged.empty:
        print("  ⚠ Henüz sonuçlanmış (oynanmış) yarış eşleşmesi yok.")
        print("    Yarışlar oynandıktan ve Stage 1 ile sonuçlar çekildikten sonra tekrar çalıştırın.")
        return

    n_races = merged["Unique_Race_ID"].nunique()
    dates = sorted(merged["Tarih"].dropna().astype(str).unique())
    print(f"  → {len(merged):,} tahmin sonuçla eşleşti | {n_races} yarış | tarihler: {dates}")

    rows = []
    # Kümülatif (tüm eşleşen yarışlar)
    for tag, pw, pt in VARIANTS:
        if merged[pw].notna().any():
            r = evaluate(merged, pw, pt)
            rows.append({"scope": "kümülatif", "Tarih": "ALL", "variant": tag,
                         "n_races": n_races, **r})
    # Günlük
    for d in dates:
        sub = merged[merged["Tarih"].astype(str) == d]
        for tag, pw, pt in VARIANTS:
            if sub[pw].notna().any():
                r = evaluate(sub, pw, pt)
                rows.append({"scope": "günlük", "Tarih": d, "variant": tag,
                             "n_races": sub["Unique_Race_ID"].nunique(), **r})

    perf = pd.DataFrame(rows)
    perf.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"  ✅ Canlı performans kaydedildi: {OUT_CSV}")

    # ── Markdown özet ──
    cum = perf[perf["scope"] == "kümülatif"]
    lines = ["# Canlı Forward-Test Performansı (Kümülatif)\n",
             f"Eşleşen yarış sayısı: **{n_races}** | Tarih aralığı: {dates[0]} → {dates[-1]}\n",
             "| Varyant | P@1 (winner) | ROI (winner, top1) | Bahis | P@1 (top3) | P@3 (top3) |",
             "|---------|--------------|--------------------|-------|------------|------------|"]
    for _, r in cum.iterrows():
        roi = f"{r['ROI_winner_top1']:+.1%}" if pd.notna(r["ROI_winner_top1"]) else "—"
        lines.append(
            f"| {r['variant']} | {r['P@1_winner']:.1%} | {roi} | "
            f"{int(r['n_bets'])} | {r['P@1_top3']:.1%} | {r['P@3_top3']:.1%} |"
        )
    lines.append("\n> `full` = ganyanlı model, `abl` = ganyansız (erken) model. "
                 "ROI yalnız kazanma bahsi içindir; final ganyanla ödenir. Kısa dönemde "
                 "yüksek varyans normaldir — anlamlı sonuç için çok sayıda yarış gerekir.")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  ✅ Özet: {OUT_MD}")
    print("\n" + "\n".join(lines[3:len(lines)-1]))


if __name__ == "__main__":
    main()
