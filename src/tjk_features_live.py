#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  TJK CANLI FEATURE ÜRETİCİ — Program satırları için (yalnızca GEÇMİŞ veri)
================================================================================
  Yarış oynanmadan, program (`program_tablo.csv`) satırları için Stage 3 ile
  AYNI master-matrix sütunlarını üretir. Target encoding'ler geçmiş sonuçlardan
  (yaris_ana_tablo) GÜNCEL kümülatif ortalama olarak hesaplanır — Stage 3'teki
  expanding().mean().shift(1) ifadesinin "bir sonraki yarış" karşılığı.

  Çıktı, Stage 4'ün load_and_validate(df) + prepare_features() yoluna doğrudan
  beslenebilecek bir DataFrame'dir (master_feature_matrix.csv ile aynı sütunlar).
================================================================================
"""

import os
import numpy as np
import pandas as pd

# Stage 3 yardımcılarını yeniden kullan (kod tekrarı yok, birebir aynı dönüşüm)
from tjk_stage3_feature_engineering import (
    derece_to_seconds, parse_yas, parse_pist_turu, extract_at_id_from_url, EPSILON,
)

# Veri dosyaları proje kökündeki data/ klasöründe (src/ -> .. -> data)
BASE_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
YARIS_CSV  = os.path.join(BASE_DIR, "yaris_ana_tablo.csv")
STATIK_CSV = os.path.join(BASE_DIR, "atlar_statik_tablo.csv")
IDMAN_CSV  = os.path.join(BASE_DIR, "idmanlar_tablo.csv")

# master_feature_matrix.csv ile aynı sütun düzeni (Stage 4'ün beklediği)
MASTER_COLS = [
    "Unique_Race_ID", "Tarih", "Kosu_ID", "Sehir", "at_id", "At_Adi",
    "Siralama", "Is_Winner", "Is_Top3",
    "Yas_Sayi", "Siklet_Sayi", "Start_Sayi", "Ganyan_Sayi", "Handikap_Puani",
    "Pist_Turu",
    "Derece_400m_sn", "Derece_600m_sn", "Derece_800m_sn", "Idman_Yaris_Arasi_Gun",
    "Jokey_Win_Rate", "Jokey_Top3_Rate",
    "Antrenor_Win_Rate", "Antrenor_Top3_Rate",
    "Baba_Win_Rate", "Baba_Top3_Rate",
    "Anne_Win_Rate", "Anne_Top3_Rate",
    "BabaAnne_Win_Rate", "BabaAnne_Top3_Rate",
    "Relative_Handikap", "Relative_Siklet", "Relative_Yas", "Yaris_At_Sayisi",
]


# ─────────────────────────────────────────────────────────────────────────────
#  GEÇMİŞTEN ENCODING LOOKUP'LARI
# ─────────────────────────────────────────────────────────────────────────────
def _build_encoding_lookups():
    """
    yaris_ana_tablo (+ statik) üzerinden her grup için GÜNCEL kümülatif oranları
    (Win_Rate, Top3_Rate) ve global ortalamaları döndürür.
    """
    yar = pd.read_csv(YARIS_CSV, encoding="utf-8-sig")
    yar["Tarih"] = pd.to_datetime(yar["Tarih"], format="%d.%m.%Y", errors="coerce")
    yar["Siralama"] = pd.to_numeric(yar["Siralama"], errors="coerce")
    yar["Is_Winner"] = (yar["Siralama"] == 1).astype(int)
    yar["Is_Top3"]   = (yar["Siralama"] <= 3).astype(int)
    yar["at_id"] = yar["At_URL"].apply(extract_at_id_from_url)

    # Statik join → Baba / Anne
    stat = pd.read_csv(STATIK_CSV, encoding="utf-8-sig")
    stat["at_id"] = pd.to_numeric(stat["at_id"], errors="coerce")
    stat_slim = stat.dropna(subset=["at_id"]).drop_duplicates("at_id")[["at_id", "Baba", "Anne"]]
    yar = yar.merge(stat_slim, on="at_id", how="left")
    yar["Baba_Anne_Key"] = yar["Baba"].astype(str) + "_x_" + yar["Anne"].astype(str)

    g_win  = float(yar["Is_Winner"].mean())
    g_top3 = float(yar["Is_Top3"].mean())

    def _rate(col):
        grp = yar.groupby(col)
        return grp["Is_Winner"].mean(), grp["Is_Top3"].mean()

    lookups = {}
    for key, src in [("Jokey", "Jokey_Adi"), ("Antrenor", "Antrenor_Adi"),
                     ("Baba", "Baba"), ("Anne", "Anne"), ("BabaAnne", "Baba_Anne_Key")]:
        win, top3 = _rate(src)
        lookups[key] = {"win": win, "top3": top3, "src": src}

    return lookups, g_win, g_top3


def _apply_encoding(prog, src_series, rate_series, global_val):
    """prog satırlarına bir grup oranını eşler; bilinmeyene global ortalama."""
    return src_series.map(rate_series).fillna(global_val).astype(float)


# ─────────────────────────────────────────────────────────────────────────────
#  İDMAN (galop) son dereceleri — bugünden ÖNCE (merge_asof backward)
# ─────────────────────────────────────────────────────────────────────────────
def _attach_idman(prog):
    """Her program satırına, yarış tarihinden ÖNCEKİ en yakın idman derecelerini ekler."""
    idman = pd.read_csv(IDMAN_CSV, encoding="utf-8-sig")
    idman["Idman_Tarihi"] = pd.to_datetime(idman["Idman_Tarihi"], format="%d.%m.%Y", errors="coerce")
    idman["at_id"] = pd.to_numeric(idman["at_id"], errors="coerce")
    idman = idman.dropna(subset=["at_id", "Idman_Tarihi"])

    for col in ["Derece_400m", "Derece_600m", "Derece_800m"]:
        if col in idman.columns:
            idman[col + "_sn"] = idman[col].apply(derece_to_seconds)
        else:
            idman[col + "_sn"] = np.nan

    keep = ["at_id", "Idman_Tarihi", "Derece_400m_sn", "Derece_600m_sn", "Derece_800m_sn"]
    idman_slim = idman[keep].sort_values(["Idman_Tarihi"]).reset_index(drop=True)

    prog_sorted = prog.sort_values("Tarih").reset_index()  # 'index' = orijinal sıra
    merged = pd.merge_asof(
        prog_sorted,
        idman_slim.rename(columns={"Idman_Tarihi": "Son_Idman_Tarihi"}).sort_values("Son_Idman_Tarihi"),
        left_on="Tarih", right_on="Son_Idman_Tarihi",
        by="at_id", direction="backward",
    )
    merged["Idman_Yaris_Arasi_Gun"] = (merged["Tarih"] - merged["Son_Idman_Tarihi"]).dt.days
    merged = merged.set_index("index").sort_index()
    for c in ["Derece_400m_sn", "Derece_600m_sn", "Derece_800m_sn", "Idman_Yaris_Arasi_Gun"]:
        prog[c] = merged[c].values
    return prog


# ─────────────────────────────────────────────────────────────────────────────
#  ANA FONKSİYON
# ─────────────────────────────────────────────────────────────────────────────
def build_live_features(program_df: pd.DataFrame) -> pd.DataFrame:
    """
    program_df (program_tablo.csv satırları) → master_feature_matrix sütunlarıyla df.
    Beklenen program sütunları: Tarih, Sehir, Kosu_ID, At_Adi, At_URL, Yas, Siklet,
    Start, Jokey_Adi, Antrenor_Adi, Ganyan (opsiyonel), Pist_Durumu.
    """
    p = program_df.copy()

    # ── Tip dönüşümleri ──
    if not np.issubdtype(p["Tarih"].dtype, np.datetime64):
        p["Tarih"] = pd.to_datetime(p["Tarih"], format="%d.%m.%Y", errors="coerce")
    p["at_id"]       = p["At_URL"].apply(extract_at_id_from_url)
    p["Yas_Sayi"]    = p["Yas"].apply(parse_yas)
    p["Siklet_Sayi"] = pd.to_numeric(p["Siklet"], errors="coerce")
    p["Start_Sayi"]  = pd.to_numeric(p["Start"], errors="coerce")
    p["Ganyan_Sayi"] = pd.to_numeric(p.get("Ganyan"), errors="coerce") if "Ganyan" in p.columns else np.nan
    p["Pist_Turu"]   = p["Pist_Durumu"].apply(parse_pist_turu) if "Pist_Durumu" in p.columns else np.nan
    p = p.dropna(subset=["at_id"]).copy()
    p["at_id"] = p["at_id"].astype(int)

    p["Unique_Race_ID"] = (
        p["Tarih"].dt.strftime("%Y%m%d") + "_" +
        p["Sehir"].astype(str).str.strip() + "_" + p["Kosu_ID"].astype(str)
    )

    # ── Statik (Handikap, Baba, Anne) ──
    stat = pd.read_csv(STATIK_CSV, encoding="utf-8-sig")
    stat["at_id"] = pd.to_numeric(stat["at_id"], errors="coerce")
    stat["Handikap_Puani"] = pd.to_numeric(stat["Handikap_Puani"], errors="coerce")
    stat_slim = stat.dropna(subset=["at_id"]).drop_duplicates("at_id")[
        ["at_id", "Handikap_Puani", "Baba", "Anne"]
    ]
    p = p.merge(stat_slim, on="at_id", how="left")
    p["Baba_Anne_Key"] = p["Baba"].astype(str) + "_x_" + p["Anne"].astype(str)

    # ── Target encoding (geçmişten) ──
    lookups, g_win, g_top3 = _build_encoding_lookups()
    src_map = {
        "Jokey":    p["Jokey_Adi"],
        "Antrenor": p["Antrenor_Adi"],
        "Baba":     p["Baba"],
        "Anne":     p["Anne"],
        "BabaAnne":  p["Baba_Anne_Key"],
    }
    for key, src_series in src_map.items():
        p[f"{key}_Win_Rate"]  = _apply_encoding(p, src_series, lookups[key]["win"],  g_win)
        p[f"{key}_Top3_Rate"] = _apply_encoding(p, src_series, lookups[key]["top3"], g_top3)

    # ── İdman dereceleri ──
    p = _attach_idman(p)

    # ── Göreceli özellikler (koşu-içi) ──
    p["Yaris_At_Sayisi"] = p.groupby("Unique_Race_ID")["at_id"].transform("count")
    for src, tgt in [("Handikap_Puani", "Relative_Handikap"),
                     ("Siklet_Sayi", "Relative_Siklet"),
                     ("Yas_Sayi", "Relative_Yas")]:
        race_mean = p.groupby("Unique_Race_ID")[src].transform("mean")
        p[tgt] = p[src] / (race_mean + EPSILON)

    # ── Placeholder hedef/sonuç sütunları (canlıda bilinmiyor) ──
    p["Siralama"]  = np.nan
    p["Is_Winner"] = 0
    p["Is_Top3"]   = 0

    # ── master_feature_matrix sütun düzenine getir ──
    for c in MASTER_COLS:
        if c not in p.columns:
            p[c] = np.nan
    out = p[MASTER_COLS].copy()
    return out


def main():
    """Standalone: program_tablo.csv → master_live.csv (test/inceleme amaçlı)."""
    prog_csv = os.path.join(BASE_DIR, "program_tablo.csv")
    if not os.path.isfile(prog_csv):
        raise SystemExit(f"[HATA] {prog_csv} yok. Önce: python tjk_stage5_live_program.py")
    prog = pd.read_csv(prog_csv, encoding="utf-8-sig")
    out = build_live_features(prog)
    out_path = os.path.join(BASE_DIR, "master_live.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✓ {len(out):,} canlı satır için feature üretildi → {out_path}")


if __name__ == "__main__":
    main()
