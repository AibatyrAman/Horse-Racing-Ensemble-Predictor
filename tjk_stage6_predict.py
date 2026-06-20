#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  TJK AŞAMA 6 — CANLI TAHMİN (yarış oynanmadan)
================================================================================
  program_tablo.csv (koşacak atlar) → tjk_features_live ile feature →
  production modellerle (HEM tam HEM ablation) Is_Winner & Is_Top3 olasılıkları.

  Imputation/encoding GEÇMİŞ veride fit edilip canlıya uygulanır (Stage 4'ün
  prepare_fold'u yeniden kullanılır) → eğitim/çıkarım paritesi garanti.

  Çıktılar:
      predictions_log.csv     – birikimli tahmin günlüğü (forward-test kaydı)
      predictions_<date>.md   – o günün okunabilir tahmin tablosu

  Kullanım:
      python tjk_stage6_predict.py            # program_tablo.csv'deki tüm tarihler
      python tjk_stage6_predict.py --date 21.06.2026
================================================================================
"""

import os
import sys
import glob
import json
import pickle
from datetime import datetime
import numpy as np
import pandas as pd

import tjk_stage4_modeling as s4
import tjk_features_live as fl
from tjk_stage3_feature_engineering import extract_at_id_from_url

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROGRAM_CSV = os.path.join(BASE_DIR, "program_tablo.csv")
PRED_LOG    = os.path.join(BASE_DIR, "predictions_log.csv")
MODEL_DIR   = s4.MODEL_DIR
TARGETS     = ["Is_Winner", "Is_Top3"]


# ─────────────────────────────────────────────────────────────────────────────
def load_production_model(target, ablation=False):
    """Registry'den hedef için en yeni production .pkl'i yükler."""
    reg_name = "production_registry_ablation.json" if ablation else "production_registry.json"
    reg_path = MODEL_DIR / reg_name
    if not reg_path.exists():
        return None, None, None
    with open(reg_path, encoding="utf-8") as f:
        reg = json.load(f)
    if target not in reg:
        return None, None, None
    name   = reg[target]["model_name"]
    suffix = "_ablation" if ablation else ""
    files  = glob.glob(str(MODEL_DIR / f"{name}_{target}{suffix}_*.pkl"))
    if not ablation:  # tam-model glob'undan ablation dosyalarını dışla
        files = [f for f in files if "_ablation_" not in os.path.basename(f)]
    if not files:
        return None, None, None
    newest = max(files, key=os.path.getmtime)
    with open(newest, "rb") as f:
        model = pickle.load(f)
    return model, name, os.path.basename(newest)


def build_inference_matrix(live_master):
    """Canlı feature matrisini eğitim verisiyle aynı yoldan geçirip imputed X döndürür."""
    hist_df = s4.load_and_validate()                      # geçmiş (fit kaynağı)
    live_df = s4.load_and_validate(live_master.copy())    # canlı (aynı engineering)

    hist_X, feat_hist = s4.prepare_features(hist_df)
    live_X, feat_live = s4.prepare_features(live_df)

    # Feature kümesi aynı olmalı; güvenli hizalama
    feats = [c for c in feat_hist if c in feat_live]
    hist_X, live_X = hist_X[feats], live_X[feats]

    # Imputation: GEÇMİŞTE fit, CANLIYA uygula (prepare_fold'u yeniden kullan)
    combined_df = pd.concat([hist_df, live_df], ignore_index=True)
    combined_X  = pd.concat([hist_X, live_X], ignore_index=True)
    hist_idx = np.arange(len(hist_df))
    live_idx = np.arange(len(hist_df), len(hist_df) + len(live_df))
    _, live_X_imp = s4.prepare_fold(combined_X, combined_df, hist_idx, live_idx)
    return live_df.reset_index(drop=True), live_X_imp.reset_index(drop=True), feats


def _race_rank(df, prob_col):
    """Yarış içi olasılık sıralaması (1 = en yüksek olasılık)."""
    return df.groupby("Unique_Race_ID")[prob_col].rank(method="first", ascending=False)


def main():
    # ── Argümanlar ──
    date_filter = None
    if "--date" in sys.argv:
        date_filter = sys.argv[sys.argv.index("--date") + 1]

    if not os.path.isfile(PROGRAM_CSV):
        raise SystemExit(f"[HATA] {PROGRAM_CSV} yok. Önce: python tjk_stage5_live_program.py")

    prog = pd.read_csv(PROGRAM_CSV, encoding="utf-8-sig")
    if date_filter:
        prog = prog[prog["Tarih"].astype(str) == date_filter].copy()
    if prog.empty:
        raise SystemExit("[HATA] Program boş (tarih filtresi sonrası kayıt yok).")

    print(f"  → {len(prog):,} program satırı için canlı feature üretiliyor...")
    live_master = fl.build_live_features(prog)
    live_df, X, feats = build_inference_matrix(live_master)
    print(f"  → {len(X):,} satır × {len(feats)} feature hazır.")

    out = live_df[["Tarih", "Sehir", "Kosu_ID", "Unique_Race_ID", "at_id",
                   "At_Adi", "Ganyan_Sayi"]].copy()

    # ── Jokey adını programdan geri ekle (okunabilirlik) ──
    pm = prog.copy()
    pm["at_id"] = pm["At_URL"].apply(extract_at_id_from_url)
    pm = pm.dropna(subset=["at_id"]); pm["at_id"] = pm["at_id"].astype(int)
    pm["Unique_Race_ID"] = (
        pd.to_datetime(pm["Tarih"], format="%d.%m.%Y", errors="coerce").dt.strftime("%Y%m%d")
        + "_" + pm["Sehir"].astype(str).str.strip() + "_" + pm["Kosu_ID"].astype(str)
    )
    out = out.merge(pm[["Unique_Race_ID", "at_id", "Jokey_Adi"]].drop_duplicates(),
                    on=["Unique_Race_ID", "at_id"], how="left")

    # ── Koşu saatini programdan geri ekle (yarış başına; okunabilirlik) ──
    if "Kosu_Saati" in pm.columns:
        saat_map = pm[["Unique_Race_ID", "Kosu_Saati"]].dropna(subset=["Kosu_Saati"]) \
                     .drop_duplicates("Unique_Race_ID")
        out = out.merge(saat_map, on="Unique_Race_ID", how="left")
    else:
        out["Kosu_Saati"] = np.nan

    # ── Tahminler: hedef × varyant ──
    any_model = False
    for target in TARGETS:
        for ablation, tag in [(False, "full"), (True, "abl")]:
            model, name, fname = load_production_model(target, ablation=ablation)
            short = "winner" if target == "Is_Winner" else "top3"
            col = f"prob_{short}_{tag}"
            if model is None:
                out[col] = np.nan
                print(f"  ⚠ {target} [{tag}] modeli bulunamadı (registry/.pkl yok) — atlandı.")
                continue
            any_model = True
            # Her modele KENDİ beklediği feature setini ver (ablation 33, tam 36).
            exp = getattr(model, "feature_names_in_", None)
            exp = list(exp) if exp is not None else feats
            out[col] = model.predict_proba(X.reindex(columns=exp))[:, 1]
            out[f"rank_{short}_{tag}"] = _race_rank(out, col).astype("Int64")
            print(f"  ✓ {target:10s} [{tag:4s}] ← {name} ({fname})")

    if not any_model:
        raise SystemExit("[HATA] Hiç production modeli yüklenemedi. Önce Stage 4'ü "
                         "(ve --ablation) çalıştırın.")

    out.insert(0, "run_ts", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # ── predictions_log.csv'ye ekle (aynı yarış-at için en yeni tahmin kalsın) ──
    if os.path.isfile(PRED_LOG):
        old = pd.read_csv(PRED_LOG, encoding="utf-8-sig")
        combined = pd.concat([old, out], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Unique_Race_ID", "at_id"], keep="last")
    else:
        combined = out
    combined.to_csv(PRED_LOG, index=False, encoding="utf-8-sig")
    print(f"\n  ✅ Tahmin günlüğü güncellendi: {PRED_LOG} ({len(out)} yeni satır)")

    # ── Okunabilir günlük tablo (her yarışın favori tahmini) ──
    write_daily_markdown(out)


def write_daily_markdown(out):
    dates = sorted(out["Tarih"].dropna().astype(str).unique())
    for d in dates:
        sub = out[out["Tarih"].astype(str) == d]
        safe = str(d).replace(".", "").replace("/", "")
        md_path = os.path.join(BASE_DIR, f"predictions_{safe}.md")
        lines = [f"# Günlük Tahmin — {d}\n",
                 "> Her yarış için modelin **1. sıra (en yüksek olasılık)** tahmini. "
                 "`full` = ganyanlı, `abl` = ganyansız (erken) model.\n",
                 "| Yarış | At (full→winner) | P(win) full | At (abl→winner) | P(win) abl | Ganyan |",
                 "|-------|------------------|-------------|-----------------|------------|--------|"]
        for rid, g in sub.groupby("Unique_Race_ID"):
            def top(col):
                if col not in g or g[col].isna().all():
                    return ("—", float("nan"))
                r = g.loc[g[col].idxmax()]
                return (r["At_Adi"], r[col])
            af, pf = top("prob_winner_full")
            aa, pa = top("prob_winner_abl")
            fav = g.loc[g["Ganyan_Sayi"].idxmin(), "At_Adi"] if g["Ganyan_Sayi"].notna().any() else "—"
            kosu = str(g["Kosu_ID"].iloc[0])
            pf_s = f"{pf:.1%}" if pf == pf else "—"
            pa_s = f"{pa:.1%}" if pa == pa else "—"
            lines.append(f"| {kosu} | {af} | {pf_s} | {aa} | {pa_s} | {fav} |")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  ✓ Günlük tablo: {md_path}")


if __name__ == "__main__":
    main()
