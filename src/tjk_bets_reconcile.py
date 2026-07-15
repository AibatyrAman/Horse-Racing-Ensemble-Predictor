#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  TJK BAHİS TAKİBİ — günlük önerilerin gerçek sonuçlarla eşleştirilmesi
================================================================================
  outputs/bets_<date>.csv (Stage 8 önerileri) × yaris_ana_tablo.csv (sonuçlar)
  → data/bets_track.csv  (her koşuda SIFIRDAN kurulur — stage7 deseni, idempotent)

  Kurallar candidate_bets'teki hit mantığının birebir aynısı:
    Ganyan        → önerilen at 1. bitirdi mi
    Plase         → önerilen at ilk 3'te mi
    İkili         → ilk 2 (sırasız küme)
    Sıralı İkili  → ilk 2 (sıralı)
    Üçlü          → ilk 3 (sıralı)
    Tabela        → ilk 4 (sıralı)

  Gerçek ödeme: Ganyan → kazananın kesin Ganyan oranı (yaris_ana_tablo).
  Egzotikler → data/payouts_tablo.csv varsa oradan (Kosu_ID + bahis etiketi);
  yoksa yalnız tuttu/tutmadı kaydedilir (profit boş kalır).

  Kullanım: python tjk_bets_reconcile.py
================================================================================
"""
import glob
import os

import numpy as np
import pandas as pd

from tjk_stage3_feature_engineering import extract_at_id_from_url, normalize_sehir

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, "data")
OUT_DIR    = os.path.join(ROOT, "outputs")
YARIS_CSV  = os.path.join(DATA_DIR, "yaris_ana_tablo.csv")
PAYOUTS_CSV = os.path.join(DATA_DIR, "payouts_tablo.csv")
TRACK_CSV  = os.path.join(DATA_DIR, "bets_track.csv")

# payouts_tablo 'Bahis' etiketi (TJK sayfa metni) → bets_track bet_type
_PAYOUT_LABELS = [
    ("SIRALI İKİLİ", "Sıralı İkili"), ("SIRALI IKILI", "Sıralı İkili"),
    ("İKİLİ", "İkili"), ("IKILI", "İkili"),
    ("ÜÇLÜ", "Üçlü"), ("UCLU", "Üçlü"),
    ("TABELA", "Tabela"), ("PLASE", "Plase"), ("GANYAN", "Ganyan"),
]


def load_results():
    """Unique_Race_ID → (bitiş sırasındaki at_id listesi, kazanan Ganyan).

    Anahtar üretimi tjk_stage7_reconcile.load_actuals ile birebir aynı;
    ek olarak bitiş sırası listesi taşınır.
    """
    y = pd.read_csv(YARIS_CSV, encoding="utf-8-sig", low_memory=False)
    y["Tarih_dt"] = pd.to_datetime(y["Tarih"], format="%d.%m.%Y", errors="coerce")
    y["Siralama_num"] = pd.to_numeric(y["Siralama"], errors="coerce")
    y["at_id"] = y["At_URL"].apply(extract_at_id_from_url)
    y = y.dropna(subset=["at_id", "Tarih_dt", "Siralama_num"])
    y["at_id"] = y["at_id"].astype(int)
    y["Unique_Race_ID"] = (
        y["Tarih_dt"].dt.strftime("%Y%m%d") + "_" +
        y["Sehir"].apply(normalize_sehir) + "_" + y["Kosu_ID"].astype(str)
    )
    y["Ganyan_num"] = pd.to_numeric(y["Ganyan"], errors="coerce")

    results = {}
    for rid, g in y.groupby("Unique_Race_ID"):
        g = g.sort_values("Siralama_num")
        order = g["at_id"].tolist()
        win_ganyan = g["Ganyan_num"].iloc[0] if len(g) else np.nan
        results[rid] = (order, win_ganyan)
    return results


def load_payouts():
    """Kosu_ID → {bet_type: ödeme (1 birim başına)}. Dosya yoksa boş."""
    if not os.path.isfile(PAYOUTS_CSV):
        return {}
    try:
        p = pd.read_csv(PAYOUTS_CSV, encoding="utf-8-sig")
    except Exception:
        return {}
    out = {}
    for _, r in p.iterrows():
        label = str(r.get("Bahis", "")).upper()
        bet_type = next((bt for pat, bt in _PAYOUT_LABELS if pat in label), None)
        tutar = pd.to_numeric(str(r.get("Tutar", "")).replace(",", "."), errors="coerce")
        if bet_type is None or pd.isna(tutar):
            continue
        out.setdefault(str(r.get("Kosu_ID")), {})[bet_type] = float(tutar)
    return out


def bet_hit(bet_type, at_ids, order):
    """candidate_bets ile aynı kurallar. order = bitiş sırasında at_id listesi."""
    if bet_type == "Ganyan":
        return len(order) >= 1 and at_ids[0] == order[0]
    if bet_type == "Plase":
        return len(order) >= 3 and at_ids[0] in order[:3]
    if bet_type == "İkili":
        return len(order) >= 2 and set(at_ids) == set(order[:2])
    if bet_type == "Sıralı İkili":
        return len(order) >= 2 and at_ids == order[:2]
    if bet_type == "Üçlü":
        return len(order) >= 3 and at_ids == order[:3]
    if bet_type == "Tabela":
        return len(order) >= 4 and at_ids == order[:4]
    return None


def main():
    bet_files = sorted(glob.glob(os.path.join(OUT_DIR, "bets_*.csv")))
    if not bet_files:
        print("  (outputs/bets_*.csv yok — takip edilecek öneri bulunamadı)")
        return
    frames = []
    for f in bet_files:
        try:
            d = pd.read_csv(f, encoding="utf-8-sig")
            if not d.empty:
                frames.append(d)
        except Exception as e:
            print(f"  ⚠ {os.path.basename(f)} okunamadı: {e}")
    if not frames:
        print("  (öneri dosyaları boş)")
        return
    bets = pd.concat(frames, ignore_index=True)
    # Aynı gün yeniden üretilmiş öneriler: son hali geçerli
    bets = bets.drop_duplicates(subset=["Unique_Race_ID", "bet_type", "at_ids"],
                                keep="last")

    results = load_results() if os.path.isfile(YARIS_CSV) else {}
    payouts = load_payouts()

    rows = []
    for _, b in bets.iterrows():
        at_ids = [int(x) for x in str(b["at_ids"]).split("|") if x.strip().isdigit()]
        rid = str(b["Unique_Race_ID"])
        stake = float(b["stake"]) if pd.notna(b["stake"]) else np.nan
        status, realized_odds, profit = "pending", np.nan, np.nan
        if rid in results and at_ids:
            order, win_ganyan = results[rid]
            hit = bet_hit(b["bet_type"], at_ids, order)
            if hit is not None:
                status = "won" if hit else "lost"
                if b["bet_type"] == "Ganyan":
                    realized_odds = win_ganyan if hit else np.nan
                    if pd.notna(stake):
                        profit = (stake * (win_ganyan - 1.0)
                                  if hit and pd.notna(win_ganyan) else -stake)
                else:
                    po = payouts.get(str(b["Kosu_ID"]), {}).get(b["bet_type"])
                    realized_odds = po if hit and po is not None else np.nan
                    if not hit and pd.notna(stake):
                        profit = -stake
                    elif hit and po is not None and pd.notna(stake):
                        profit = stake * (po - 1.0)
        rows.append({
            "Tarih": b["Tarih"], "Sehir": b["Sehir"], "Kosu_ID": b["Kosu_ID"],
            "Unique_Race_ID": rid, "bet_type": b["bet_type"],
            "horses": b["horses"], "at_ids": b["at_ids"],
            "p_model": b["p_model"], "payout_est": b["payout_est"],
            "ev": b["ev"], "stake": stake, "status": status,
            "realized_odds": realized_odds, "profit": profit,
        })

    track = pd.DataFrame(rows).sort_values(
        ["Tarih", "Sehir", "Kosu_ID"], ascending=[False, True, True])
    track.to_csv(TRACK_CSV, index=False, encoding="utf-8-sig")

    resolved = track[track["status"] != "pending"]
    won = (resolved["status"] == "won").sum()
    gany = resolved[resolved["bet_type"] == "Ganyan"]
    gany_profit = gany["profit"].sum() if not gany.empty else 0.0
    gany_stake = gany["stake"].sum() if not gany.empty else 0.0
    roi_txt = f"{gany_profit / gany_stake:+.1%}" if gany_stake > 0 else "—"
    print(f"  ✅ Bahis takibi: {len(track)} öneri "
          f"({len(resolved)} sonuçlandı, {won} tuttu; "
          f"Ganyan gerçek ROI: {roi_txt}) → {TRACK_CSV}")


if __name__ == "__main__":
    main()
