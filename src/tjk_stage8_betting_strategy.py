#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  TJK AŞAMA 8 — BAHİS STRATEJİSİ & KASA OPTİMİZASYONU
================================================================================
  Modelin per-at olasılıklarını (Harville/Plackett-Luce ile) egzotik bahis
  kombinasyon olasılıklarına çevirir; piyasa-ima ödeme ile beklenen değer (EV)
  hesaplar; pozitif-EV bahisleri seçer; sabit kasayla **flat** ve **fractional
  Kelly** stratejilerini kronolojik simüle eder.

  İki mod:
    --backtest          oof_predictions.csv (sızıntısız geçmiş) üzerinde tam
                        backtest → betting_strategy_backtest.csv,
                        reports/betting_strategy_summary.md, reports/bankroll_curve.png
    --date YYYY-MM-DD   predictions_log.csv'den o güne öneri → bets_<date>.md

  DÜRÜSTLÜK: Egzotik geçmiş ödemeleri veride YOK. Backtest'te ödeme piyasa-ima
  ile tahmin edilir (payout ≈ (1−takeout)/P_market). Bu yüzden backtest bir
  "göreli-edge" simülasyonudur (model piyasayı doğru yönde yenebiliyor mu?),
  literal TL P&L değil. Ganyan bahsi istisna: gerçek Ganyan oranı kullanılır.
  Forward modda payouts_tablo.csv (gerçek ödemeler) varsa onlar kullanılır.

  Kullanım:
    python tjk_stage8_betting_strategy.py --backtest
    python tjk_stage8_betting_strategy.py --date 2026-06-20
================================================================================
"""
import os
import argparse
import numpy as np
import pandas as pd

import tjk_betting as bet

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/ -> kök
DATA_DIR  = os.path.join(ROOT, "data")
OUT_DIR   = os.path.join(ROOT, "outputs")
OOF_CSV   = os.path.join(DATA_DIR, "oof_predictions.csv")
PRED_LOG  = os.path.join(DATA_DIR, "predictions_log.csv")
REPORTS   = os.path.join(ROOT, "reports")

# ── Yapılandırma (varsayılanlar; CLI ile geçersiz kılınabilir) ─────────────────
INITIAL_BANKROLL = 1000.0   # başlangıç kasası (TL)
FLAT_FRAC        = 0.01     # flat: her bahis = başlangıç kasasının %1'i (sabit)
KELLY_FRAC       = 0.25     # ¼-Kelly
KELLY_CAP        = 0.05     # tek bahiste kasanın en çok %5'i
EV_THRESHOLD     = 0.05     # bahis için gereken min. edge (EV/stake)
LAMBDA           = 0.85     # favori-uzunatış kalibrasyon üssü (Lo-Bacon-Shor)
TOPN             = 4        # kombinasyonlar yalnız modele göre ilk-N at arasından

# ── Güvenilirlik korumaları (nadir kombinasyonların piyasa-ima ödemesi patlamasın) ──
MIN_COMBO_PROB   = 0.02     # egzotikte modelin kombinasyon olasılığı en az %2 olmalı
MAX_PAYOUT_EXOTIC = 50.0    # egzotik ödeme tahmini tavanı (üstü güvenilmez → bahis yok)
MAX_BETS_PER_RACE = 2       # bir koşuda en çok kaç bahis (en yüksek EV'liler)

# Bahis türü → takeout (TJK yaklaşık komisyon oranları)
TAKEOUT = {
    "Ganyan":       0.19,
    "Plase":        0.20,
    "İkili":        0.25,
    "Sıralı İkili": 0.25,
    "Üçlü":         0.27,
    "Tabela":       0.30,
}
# Çoklu-koşu (deneysel) takeout
TAKEOUT_PICKN = 0.30


# ──────────────────────────────────────────────────────────────────────────────
#  VERİ YÜKLEME
# ──────────────────────────────────────────────────────────────────────────────
def load_probs(mode, variant="full"):
    """
    Olasılık kaynağını döndürür (ortak şema):
      cols: Unique_Race_ID, Tarih, Sehir, Kosu_ID, at_id, At_Adi,
            model_win, model_top3, Ganyan, Siralama
    mode='backtest' → oof_predictions.csv ; mode='live' → predictions_log.csv
    """
    if mode == "backtest":
        if not os.path.isfile(OOF_CSV):
            raise FileNotFoundError(
                f"{OOF_CSV} yok. Önce: python tjk_stage4_modeling.py --dump-oof")
        d = pd.read_csv(OOF_CSV, encoding="utf-8-sig")
        out = pd.DataFrame({
            "Unique_Race_ID": d["Unique_Race_ID"],
            "Tarih":   pd.to_datetime(d["Tarih"], errors="coerce"),
            "Sehir":   d["Sehir"].astype(str),
            "Kosu_ID": d["Kosu_ID"],
            "at_id":   d["at_id"],
            "At_Adi":  d["At_Adi"].astype(str),
            "model_win":  pd.to_numeric(d["oof_prob_winner"], errors="coerce"),
            "model_top3": pd.to_numeric(d["oof_prob_top3"], errors="coerce"),
            "Ganyan":  pd.to_numeric(d["Ganyan_Sayi"], errors="coerce"),
            "Siralama": pd.to_numeric(d["Siralama"], errors="coerce"),
        })
    else:  # live
        if not os.path.isfile(PRED_LOG):
            raise FileNotFoundError(f"{PRED_LOG} yok. Önce tahmin üret (Stage 6).")
        d = pd.read_csv(PRED_LOG, encoding="utf-8-sig")
        w_col = f"prob_winner_{variant}"
        t_col = f"prob_top3_{variant}"
        out = pd.DataFrame({
            "Unique_Race_ID": d["Unique_Race_ID"],
            "Tarih":   pd.to_datetime(d["Tarih"], errors="coerce"),
            "Sehir":   d["Sehir"].astype(str),
            "Kosu_ID": d["Kosu_ID"],
            "at_id":   d["at_id"],
            "At_Adi":  d["At_Adi"].astype(str),
            "model_win":  pd.to_numeric(d[w_col], errors="coerce"),
            "model_top3": pd.to_numeric(d[t_col], errors="coerce"),
            "Ganyan":  pd.to_numeric(d["Ganyan_Sayi"], errors="coerce"),
            "Siralama": np.nan,  # canlı: sonuç henüz yok
        })
        out["Kosu_Saati"] = d["Kosu_Saati"] if "Kosu_Saati" in d.columns else np.nan
    return out


def build_race(g, lam=LAMBDA):
    """
    Bir koşu grubundan strateji girdilerini hazırlar.
    Döndürür: dict (names, p_model, p_market, p_top3, odds, order, n) veya
    yetersiz/eksik veri varsa None.
    """
    g = g.reset_index(drop=True)
    if g["Ganyan"].isna().all() or g["model_win"].isna().all():
        return None
    n = len(g)
    if n < 2:
        return None

    names  = g["At_Adi"].tolist()
    odds   = g["Ganyan"].to_numpy(dtype=float)
    # Model kazanma dağılımı: koşu-içi normalize + kalibrasyon
    p_model = bet.calibrate(np.nan_to_num(g["model_win"].to_numpy(dtype=float)), lam)
    # Piyasa kazanma dağılımı: Ganyan oranlarından + kalibrasyon
    p_market = bet.calibrate(bet.market_probs(odds), lam)
    # İlk-3 (Plase) için marjinal model olasılığı
    p_top3 = np.nan_to_num(g["model_top3"].to_numpy(dtype=float))

    # Gerçek bitiş sırası (Siralama varsa): indeksler 1.,2.,3.,...
    if g["Siralama"].notna().any():
        ordered = g.dropna(subset=["Siralama"]).sort_values("Siralama")
        order = ordered.index.tolist()
    else:
        order = None  # canlı: sonuç yok

    return {"names": names, "p_model": p_model, "p_market": p_market,
            "p_top3": p_top3, "odds": odds, "order": order, "n": n}


# ──────────────────────────────────────────────────────────────────────────────
#  TEK-KOŞU ADAY BAHİSLERİ
# ──────────────────────────────────────────────────────────────────────────────
def _topn_idx(p, n):
    return list(np.argsort(p)[::-1][:n])


def candidate_bets(R, topn=TOPN, ev_threshold=EV_THRESHOLD):
    """
    Bir koşu için her bahis türünde en iyi (max-EV) kombinasyonu bulur; edge
    eşiğini geçenleri döndürür. Skorlama (order varsa) hit alanını doldurur.
    """
    pm, pq, pt3, odds = R["p_model"], R["p_market"], R["p_top3"], R["odds"]
    n = R["n"]
    cand = _topn_idx(pm, min(topn, n))
    order = R["order"]
    bets = []

    def finish_at(rank):  # rank 0-based → o sırada biten at indeksi
        return order[rank] if (order is not None and len(order) > rank) else None

    # ── Ganyan (kazanan) — GERÇEK Ganyan oranı ile ──
    i = cand[0]
    payout = odds[i]
    if np.isfinite(payout) and payout > 1.0:
        evr = bet.ev_ratio(pm[i], payout)
        hit = (finish_at(0) == i) if order is not None else None
        bets.append(_mk("Ganyan", [i], R, pm[i], None, payout, evr, hit))

    # ── Plase (ilk-3) — piyasa-ima ödeme ──
    j = int(np.argmax(pt3))
    p_mk_place = bet.place_prob(pq, j, n_places=3)
    payout = bet.implied_payout(p_mk_place, TAKEOUT["Plase"])
    evr = bet.ev_ratio(pt3[j], payout)
    hit = (order is not None and len(order) >= 3 and j in order[:3]) if order is not None else None
    bets.append(_mk("Plase", [j], R, pt3[j], p_mk_place, payout, evr, hit))

    # ── İkili (sırasız ilk-2) ──
    best = _best_combo(cand, 2, ordered=False, pm=pm, pmkt=pq, takeout=TAKEOUT["İkili"])
    if best:
        combo, p_mod, p_mk, payout, evr = best
        hit = (order is not None and len(order) >= 2 and
               set(combo) == set(order[:2])) if order is not None else None
        bets.append(_mk("İkili", list(combo), R, p_mod, p_mk, payout, evr, hit))

    # ── Sıralı İkili (sıralı ilk-2) ──
    best = _best_combo(cand, 2, ordered=True, pm=pm, pmkt=pq, takeout=TAKEOUT["Sıralı İkili"])
    if best:
        combo, p_mod, p_mk, payout, evr = best
        hit = (order is not None and len(order) >= 2 and
               tuple(combo) == tuple(order[:2])) if order is not None else None
        bets.append(_mk("Sıralı İkili", list(combo), R, p_mod, p_mk, payout, evr, hit))

    # ── Üçlü (sıralı ilk-3) ──
    if n >= 3:
        best = _best_combo(cand, 3, ordered=True, pm=pm, pmkt=pq, takeout=TAKEOUT["Üçlü"])
        if best:
            combo, p_mod, p_mk, payout, evr = best
            hit = (order is not None and len(order) >= 3 and
                   tuple(combo) == tuple(order[:3])) if order is not None else None
            bets.append(_mk("Üçlü", list(combo), R, p_mod, p_mk, payout, evr, hit))

    # ── Tabela (sıralı ilk-4) ──
    if n >= 4:
        best = _best_combo(cand, 4, ordered=True, pm=pm, pmkt=pq, takeout=TAKEOUT["Tabela"])
        if best:
            combo, p_mod, p_mk, payout, evr = best
            hit = (order is not None and len(order) >= 4 and
                   tuple(combo) == tuple(order[:4])) if order is not None else None
            bets.append(_mk("Tabela", list(combo), R, p_mod, p_mk, payout, evr, hit))

    # ── Güvenilirlik filtreleri ──
    sel = []
    for b in bets:
        if b["ev_ratio"] is None or b["ev_ratio"] < ev_threshold:
            continue
        if b["payout"] <= 1.0:                       # kazanca imkân yok
            continue
        if b["bet_type"] != "Ganyan":               # Ganyan gerçek oran → muaf
            # Egzotik/Plase: nadir & güvenilmez tahminleri ele
            if b["p_model"] < MIN_COMBO_PROB:
                continue
            if b["payout"] > MAX_PAYOUT_EXOTIC:
                continue
        sel.append(b)
    # Koşu başına en çok MAX_BETS_PER_RACE (en yüksek EV'liler)
    sel.sort(key=lambda x: x["ev_ratio"], reverse=True)
    return sel[:MAX_BETS_PER_RACE]


def _best_combo(cand, k, ordered, pm, pmkt, takeout):
    """cand içinden k'lık en yüksek EV kombinasyonu (model olasılığına göre)."""
    from itertools import permutations, combinations
    gen = permutations(cand, k) if ordered else combinations(cand, k)
    best = None
    for combo in gen:
        if ordered:
            p_mod = bet.ordered_topk_prob(pm, combo)
            p_mk  = bet.ordered_topk_prob(pmkt, combo)
        else:
            p_mod = bet.unordered_topk_prob(pm, combo)
            p_mk  = bet.unordered_topk_prob(pmkt, combo)
        if p_mk <= bet.EPSILON:
            continue
        payout = bet.implied_payout(p_mk, takeout)
        evr = bet.ev_ratio(p_mod, payout)
        if best is None or evr > best[4]:
            best = (combo, p_mod, p_mk, payout, evr)
    return best


def _mk(bet_type, combo, R, p_mod, p_mk, payout, evr, hit):
    names = "-".join(R["names"][c] for c in combo)
    return {"bet_type": bet_type, "combo": combo, "horses": names,
            "p_model": p_mod, "p_market": p_mk, "payout": payout,
            "ev_ratio": evr, "hit": hit}


# ──────────────────────────────────────────────────────────────────────────────
#  ÇOKLU-KOŞU (DENEYSEL): Çifte (2 bacak) + 3'lü Ganyan (3 bacak)
# ──────────────────────────────────────────────────────────────────────────────
def multi_race_bets(card_races, ev_threshold=EV_THRESHOLD):
    """
    Bir yarış günü-şehri (card) içindeki ardışık koşulardan banker (her bacakta
    top-1) Çifte ve 3'lü Ganyan biletleri. Yüksek varyans → ayrı raporlanır.
    card_races: kronolojik [R, ...] (her biri build_race çıktısı).
    """
    out = []
    for legs, name in ((2, "Çifte"), (3, "3'lü Ganyan")):
        for s in range(0, len(card_races) - legs + 1):
            window = card_races[s:s + legs]
            if any(w is None for w in window):
                continue
            sel = [int(np.argmax(w["p_model"])) for w in window]
            p_mod = float(np.prod([w["p_model"][i] for w, i in zip(window, sel)]))
            p_mk  = float(np.prod([w["p_market"][i] for w, i in zip(window, sel)]))
            if p_mk <= bet.EPSILON:
                continue
            payout = bet.implied_payout(p_mk, TAKEOUT_PICKN)
            evr = bet.ev_ratio(p_mod, payout)
            # hit: her bacakta seçilen at 1. mi?
            hit = None
            if all(w["order"] is not None for w in window):
                hit = all(w["order"][0] == i for w, i in zip(window, sel))
            if evr >= ev_threshold:
                horses = " | ".join(window[t]["names"][sel[t]] for t in range(legs))
                out.append({"bet_type": name, "legs": legs, "horses": horses,
                            "p_model": p_mod, "p_market": p_mk, "payout": payout,
                            "ev_ratio": evr, "hit": hit})
    return out


# ──────────────────────────────────────────────────────────────────────────────
#  KASA SİMÜLASYONU (flat vs fractional Kelly)
# ──────────────────────────────────────────────────────────────────────────────
def simulate(records, initial=INITIAL_BANKROLL, flat_frac=FLAT_FRAC,
             kelly_frac=KELLY_FRAC, kelly_cap=KELLY_CAP):
    """
    Kronolojik bahis kayıtları üzerinde iki kasa simüle eder.
    records: [{race_seq, hit, payout, p_model}, ...] (yalnız skorlanabilir, hit!=None)
    Döndürür: kasa eğrisi DataFrame + özet dict.
    """
    flat_stake = initial * flat_frac
    bank_flat = initial
    bank_kelly = initial
    curve = []
    n_bets = 0
    n_wins = 0
    staked_flat = 0.0
    ret_flat = 0.0

    for r in records:
        if r["hit"] is None:
            continue
        payout = r["payout"]
        b = payout - 1.0
        # flat
        s_flat = min(flat_stake, bank_flat)
        if s_flat > 0:
            pnl_flat = s_flat * (payout - 1.0) if r["hit"] else -s_flat
            bank_flat += pnl_flat
            staked_flat += s_flat
            ret_flat += pnl_flat
        # kelly
        f = bet.fractional_kelly(r["p_model"], b, frac=kelly_frac, cap=kelly_cap)
        s_kelly = bank_kelly * f
        if s_kelly > 0:
            pnl_kelly = s_kelly * (payout - 1.0) if r["hit"] else -s_kelly
            bank_kelly += pnl_kelly
        n_bets += 1
        n_wins += int(bool(r["hit"]))
        curve.append({"seq": n_bets, "Tarih": r.get("Tarih"),
                      "bank_flat": bank_flat, "bank_kelly": bank_kelly})

    summary = {
        "n_bets": n_bets,
        "n_wins": n_wins,
        "hit_rate": (n_wins / n_bets) if n_bets else 0.0,
        "bank_flat_final": bank_flat,
        "bank_kelly_final": bank_kelly,
        "roi_flat": (ret_flat / staked_flat) if staked_flat > 0 else 0.0,
    }
    return pd.DataFrame(curve), summary


# ──────────────────────────────────────────────────────────────────────────────
#  BACKTEST
# ──────────────────────────────────────────────────────────────────────────────
def run_backtest(lam=LAMBDA, ev_threshold=EV_THRESHOLD, initial=INITIAL_BANKROLL):
    df = load_probs("backtest")
    df = df.sort_values(["Tarih", "Sehir", "Kosu_ID"]).reset_index(drop=True)

    all_bets = []        # tek-koşu skorlanmış bahisler
    multi_bets = []      # çoklu-koşu (deneysel)
    # card = (Tarih, Sehir) → ardışık koşular
    for (tarih, sehir), card in df.groupby([df["Tarih"].dt.date, "Sehir"], sort=True):
        card_races = []
        for rid, g in card.groupby("Kosu_ID", sort=True):
            R = build_race(g, lam=lam)
            card_races.append(R)
            if R is None:
                continue
            for b in candidate_bets(R, ev_threshold=ev_threshold):
                b["Tarih"] = tarih
                b["Sehir"] = sehir
                b["Kosu_ID"] = rid
                all_bets.append(b)
        for mb in multi_race_bets(card_races, ev_threshold=ev_threshold):
            mb["Tarih"] = tarih
            mb["Sehir"] = sehir
            multi_bets.append(mb)

    bets_df = pd.DataFrame(all_bets)

    # ── (1) DÜRÜST HEADLINE: model-seçimi vs piyasa-seçimi isabet tanısı (ödemesiz) ──
    diag = edge_diagnostic(df, lam=lam)

    # ── (2) Tek GÜVENİLİR kasa: Ganyan (GERÇEK oran, dairesel değil) ──
    ganyan = [b for b in all_bets if b["bet_type"] == "Ganyan" and b["hit"] is not None]
    ganyan.sort(key=lambda r: (r["Tarih"], str(r["Sehir"]), r["Kosu_ID"]))
    g_curve, g_summary = simulate(ganyan, initial=initial)

    # ── (3) Egzotik EV tablosu — yalnız GÖSTERGE (ödeme tahmini → dairesel) ──
    by_type = _by_type_summary(bets_df)
    multi_summary = _by_type_summary(pd.DataFrame(multi_bets)) if multi_bets else pd.DataFrame()

    # Çıktılar
    os.makedirs(REPORTS, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    bt_path = os.path.join(DATA_DIR, "betting_strategy_backtest.csv")
    if not bets_df.empty:
        bets_df.assign(combo=bets_df["combo"].apply(lambda c: "-".join(map(str, c)))) \
               .to_csv(bt_path, index=False, encoding="utf-8-sig")
    _write_summary(diag, g_summary, by_type, multi_summary, lam, ev_threshold, initial,
                   n_races=df["Unique_Race_ID"].nunique())
    _plot_curve(g_curve, initial)

    print(f"\n  ✅ Backtest tamam.")
    if not diag.empty:
        print("\n  Model vs Piyasa seçim isabeti (ödemesiz, dürüst):")
        for _, r in diag.iterrows():
            print(f"    {r['Bahis']:14s} model={r['model_isabet']:.1%} "
                  f"piyasa={r['piyasa_isabet']:.1%}  (Δ {r['fark_pp']:+.1f}pp, n={int(r['n'])})")
    print(f"\n  Ganyan kasası (GERÇEK oran): {g_summary['n_bets']} bahis | "
          f"isabet={g_summary['hit_rate']:.1%} | flat ROI {g_summary['roi_flat']:+.1%}")
    print("  ⚠ Backtest iyimser — forward-test gerçek hakem (Ganyan ROI ~ -36%).")
    print(f"  → {bt_path}")
    print(f"  → {os.path.join(REPORTS, 'betting_strategy_summary.md')}")
    print(f"  → {os.path.join(REPORTS, 'bankroll_curve.png')}")
    return g_summary


# Tanı için bahis türü spesifikasyonları: (ad, k, sıralı_mı)
DIAG_SPECS = [
    ("Ganyan",       1, True),
    ("İkili",        2, False),
    ("Sıralı İkili", 2, True),
    ("Üçlü",         3, True),
    ("Tabela",       4, True),
]


def edge_diagnostic(df, lam=LAMBDA):
    """
    DÜRÜST, DAİRESEL-OLMAYAN tanı: ödeme VARSAYMADAN, modelin doğal seçimi
    (olasılığa göre ilk-k) piyasanın doğal seçiminden (favori sırasına göre ilk-k)
    daha sık mı tutuyor? Pozitif fark → model o bahis türünde değer katıyor.
    """
    recs = {name: {"m": 0, "q": 0, "n": 0} for name, _, _ in DIAG_SPECS}
    place = {"m": 0, "q": 0, "n": 0}
    for _, g in df.groupby("Unique_Race_ID"):
        R = build_race(g, lam=lam)
        if R is None or R["order"] is None:
            continue
        order, n = R["order"], R["n"]
        m_rank = list(np.argsort(R["p_model"])[::-1])
        q_rank = list(np.argsort(R["p_market"])[::-1])
        for name, k, ordered in DIAG_SPECS:
            if n < k or len(order) < k:
                continue
            m_pick, q_pick = m_rank[:k], q_rank[:k]
            if ordered:
                m_hit = tuple(m_pick) == tuple(order[:k])
                q_hit = tuple(q_pick) == tuple(order[:k])
            else:
                m_hit = set(m_pick) == set(order[:k])
                q_hit = set(q_pick) == set(order[:k])
            recs[name]["n"] += 1
            recs[name]["m"] += int(m_hit)
            recs[name]["q"] += int(q_hit)
        if n >= 3 and len(order) >= 3:
            m_place = int(np.argmax(R["p_top3"]))
            q_place = int(np.argmax(R["p_market"]))
            place["n"] += 1
            place["m"] += int(m_place in order[:3])
            place["q"] += int(q_place in order[:3])

    out = []
    for name, _, _ in DIAG_SPECS:
        r = recs[name]
        if r["n"]:
            out.append({"Bahis": name, "n": r["n"],
                        "model_isabet": r["m"] / r["n"],
                        "piyasa_isabet": r["q"] / r["n"]})
    if place["n"]:
        out.append({"Bahis": "Plase", "n": place["n"],
                    "model_isabet": place["m"] / place["n"],
                    "piyasa_isabet": place["q"] / place["n"]})
    d = pd.DataFrame(out)
    if not d.empty:
        d["fark_pp"] = (d["model_isabet"] - d["piyasa_isabet"]) * 100
    return d


def _by_type_summary(bets_df):
    if bets_df is None or bets_df.empty:
        return pd.DataFrame()
    d = bets_df[bets_df["hit"].notna()].copy()
    if d.empty:
        return pd.DataFrame()
    d["hit"] = d["hit"].astype(int)
    grp = d.groupby("bet_type").agg(
        n=("hit", "size"),
        isabet=("hit", "mean"),
        ort_payout=("payout", "mean"),
        ort_ev=("ev_ratio", "mean"),
    ).reset_index()
    # Basit ROI: Σ(hit·(payout−1) − (1−hit)) / n   (eşit pay)
    roi = []
    for bt, sub in d.groupby("bet_type"):
        pnl = (sub["hit"] * (sub["payout"] - 1.0) - (1 - sub["hit"])).sum()
        roi.append(pnl / len(sub))
    grp["roi_flat"] = roi
    return grp.sort_values("n", ascending=False)


def _write_summary(diag, g_summary, by_type, multi_summary, lam, ev_threshold, initial, n_races):
    path = os.path.join(REPORTS, "betting_strategy_summary.md")
    L = []
    L.append("# Bahis Stratejisi — Backtest Özeti\n")
    L.append(f"**Kapsam:** {n_races:,} koşu (OOF, sızıntısız) • "
             f"λ={lam} • EV eşiği={ev_threshold:+.0%}\n")

    # ── (1) DÜRÜST HEADLINE — model vs piyasa seçim isabeti (ödemesiz) ──
    L.append("\n## 1) Model vs Piyasa — seçim isabeti (ödeme VARSAYMADAN)\n")
    L.append("> En sağlam, dairesel-olmayan kanıt: modelin doğal seçimi (olasılığa göre "
             "ilk-k) piyasanın doğal seçiminden (favori sırası) daha sık mı tutuyor? "
             "Pozitif Δ → model o türde değer katıyor.\n")
    if diag is not None and not diag.empty:
        L.append("| Bahis | n | Model isabet | Piyasa isabet | Δ (pp) |")
        L.append("|-------|---|--------------|---------------|--------|")
        for _, r in diag.iterrows():
            L.append(f"| {r['Bahis']} | {int(r['n'])} | {r['model_isabet']:.1%} | "
                     f"{r['piyasa_isabet']:.1%} | {r['fark_pp']:+.1f} |")

    # ── (2) Ganyan kasası — tek GÜVENİLİR bankroll (gerçek oran) ──
    L.append("\n## 2) Ganyan kasası — GERÇEK oran (tek güvenilir bankroll)\n")
    L.append("> Yalnız Ganyan'da gerçek ödeme (oran) var → bu kasa dairesel değil. "
             "Flat = sabit pay, Kelly = ¼-Kelly (cap %5).\n")
    L.append(f"- Bahis: **{g_summary['n_bets']}** • isabet: **{g_summary['hit_rate']:.1%}**")
    L.append(f"- Flat kasa: **{initial:.0f} → {g_summary['bank_flat_final']:.0f} TL** "
             f"(ROI/stake {g_summary['roi_flat']:+.1%})")
    L.append(f"- Kelly kasa: **{initial:.0f} → {g_summary['bank_kelly_final']:.3g} TL**\n")
    L.append("> ⚠️ **Backtest İYİMSER.** Aynı modelin canlı forward-testinde Ganyan ROI "
             "**~ −36%** çıktı (18 koşu). Geçmiş OOF backtest geleceği garanti ETMEZ; "
             "gerçek hakem forward-test'tir. Kelly'nin büyük görünmesi, dairesel-olmayan "
             "ama iyimser edge'in üst üste katlanmasıdır.\n")

    # ── (3) Egzotik EV — yalnız GÖSTERGE (dairesel) ──
    L.append("\n## 3) Egzotik bahis türleri — GÖSTERGE (literal TL DEĞİL)\n")
    L.append("> ⚠️ Egzotik geçmiş ödemesi yok → ödeme piyasa-ima ile tahmin edildi "
             "`(1−takeout)/P_market`. Skorlamada da aynı tahmin kullanıldığından bu **dairesel**; "
             "ROI fantazidir. Yalnız *hangi türde sinyal var* fikri için; bankroll olarak alınmaz. "
             "Gerçek değerlendirme için yukarıdaki (1) tanısına ve forward-test'e bakın.\n")
    if not by_type.empty:
        L.append("| Tür | n | İsabet | Ort. ödeme~ | Ort. EV~ |")
        L.append("|-----|---|--------|-------------|----------|")
        for _, r in by_type.iterrows():
            L.append(f"| {r['bet_type']} | {int(r['n'])} | {r['isabet']:.1%} | "
                     f"{r['ort_payout']:.2f} | {r['ort_ev']:+.1%} |")

    if not multi_summary.empty:
        L.append("\n### Çoklu-koşu (DENEYSEL — en yüksek varyans, gösterge)\n")
        L.append("| Tür | n | İsabet | Ort. ödeme~ | Ort. EV~ |")
        L.append("|-----|---|--------|-------------|----------|")
        for _, r in multi_summary.iterrows():
            L.append(f"| {r['bet_type']} | {int(r['n'])} | {r['isabet']:.1%} | "
                     f"{r['ort_payout']:.2f} | {r['ort_ev']:+.1%} |")

    L.append("\n---\n*Araştırma/kâğıt-üzeri amaçlı. Gerçek bahis önerilmez.*")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def _plot_curve(curve, initial):
    """Ganyan (gerçek oran) kasa eğrisi — flat vs Kelly. Kelly log-eksende."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if curve is None or curve.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(curve["seq"], curve["bank_flat"], color="#1f77b4", lw=1.8)
    ax1.axhline(initial, color="gray", ls="--", lw=1)
    ax1.set_title("Ganyan — Flat (sabit pay)")
    ax1.set_xlabel("Bahis sırası"); ax1.set_ylabel("Kasa (TL)"); ax1.grid(alpha=0.3)

    ax2.plot(curve["seq"], curve["bank_kelly"], color="#ff7f0e", lw=1.8)
    ax2.axhline(initial, color="gray", ls="--", lw=1)
    ax2.set_yscale("log")
    ax2.set_title("Ganyan — ¼-Kelly (log eksen; backtest iyimser)")
    ax2.set_xlabel("Bahis sırası"); ax2.set_ylabel("Kasa (TL, log)"); ax2.grid(alpha=0.3)

    fig.suptitle("Ganyan kasa eğrisi (GERÇEK oran) — backtest iyimser; forward-test hakem")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "bankroll_curve.png"), dpi=130)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
#  GÜNLÜK ÖNERİ (canlı)
# ──────────────────────────────────────────────────────────────────────────────
def run_recommendations(date_str, variant="full", lam=LAMBDA,
                        ev_threshold=EV_THRESHOLD, bankroll=INITIAL_BANKROLL):
    df = load_probs("live", variant=variant)
    target_date = pd.to_datetime(date_str, errors="coerce").date()
    df = df[df["Tarih"].dt.date == target_date]
    if df.empty:
        print(f"  ⚠ {date_str} için tahmin bulunamadı ({PRED_LOG}).")
        return
    df = df.sort_values(["Sehir", "Kosu_ID"]).reset_index(drop=True)

    rows = []
    for (sehir, rid), g in df.groupby(["Sehir", "Kosu_ID"], sort=True):
        R = build_race(g, lam=lam)
        if R is None:
            continue
        # Koşu etiketi: ID + (varsa) saat
        kosu_lbl = str(rid)
        if "Kosu_Saati" in g.columns:
            s = g["Kosu_Saati"].dropna().astype(str)
            s = s[s.str.strip().ne("") & s.ne("nan")]
            if not s.empty:
                kosu_lbl = f"{rid} ({s.iloc[0]})"
        for b in candidate_bets(R, ev_threshold=ev_threshold):
            f = bet.fractional_kelly(b["p_model"], b["payout"] - 1.0,
                                     frac=KELLY_FRAC, cap=KELLY_CAP)
            rows.append({
                "Sehir": sehir, "Kosu": kosu_lbl, "Bahis": b["bet_type"],
                "Atlar": b["horses"], "P(model)": b["p_model"],
                "Ödeme~": b["payout"], "EV": b["ev_ratio"],
                "Pay (Kelly)": bankroll * f,
            })

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"bets_{date_str}.md")
    lines = [f"# Bahis Önerileri — {date_str}\n",
             "> Araştırma/kâğıt-üzeri. Egzotik ödemeler piyasa-ima tahminidir; "
             "gerçek bahis önerilmez.\n",
             f"**Varsayım:** λ={lam} • EV eşiği={ev_threshold:+.0%} • "
             f"kasa={bankroll:.0f} TL • Kelly={KELLY_FRAC:g}×(cap {KELLY_CAP:.0%})\n"]
    if not rows:
        lines.append("\n_Bugün için pozitif-EV (eşik üstü) bahis bulunamadı._")
    else:
        rec = pd.DataFrame(rows).sort_values(["Sehir", "Kosu", "EV"],
                                             ascending=[True, True, False])
        lines.append("\n| Şehir | Koşu | Bahis | Atlar | P(model) | Ödeme~ | EV | Pay (Kelly) |")
        lines.append("|-------|------|-------|-------|----------|--------|----|-------------|")
        for _, r in rec.iterrows():
            lines.append(f"| {r['Sehir']} | {r['Kosu']} | {r['Bahis']} | {r['Atlar']} | "
                         f"{r['P(model)']:.1%} | {r['Ödeme~']:.2f} | {r['EV']:+.1%} | "
                         f"{r['Pay (Kelly)']:.1f} TL |")
        lines.append(f"\n**Toplam önerilen bahis:** {len(rows)} • "
                     f"toplam pay: {sum(r['Pay (Kelly)'] for r in rows):.1f} TL")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  ✅ {len(rows)} öneri → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="TJK bahis stratejisi & kasa optimizasyonu")
    ap.add_argument("--backtest", action="store_true", help="OOF üzerinde tam backtest")
    ap.add_argument("--date", type=str, help="Günlük öneri (YYYY-MM-DD)")
    ap.add_argument("--variant", default="full", choices=["full", "abl"],
                    help="Canlı öneri için model varyantı")
    ap.add_argument("--lam", type=float, default=LAMBDA, help="Kalibrasyon üssü λ")
    ap.add_argument("--ev", type=float, default=EV_THRESHOLD, help="EV eşiği")
    ap.add_argument("--bankroll", type=float, default=INITIAL_BANKROLL, help="Başlangıç kasa")
    args = ap.parse_args()

    print("\n" + "█" * 70)
    print("  TJK AŞAMA 8: BAHİS STRATEJİSİ & KASA OPTİMİZASYONU")
    print("█" * 70)

    if args.backtest:
        run_backtest(lam=args.lam, ev_threshold=args.ev, initial=args.bankroll)
    elif args.date:
        run_recommendations(args.date, variant=args.variant, lam=args.lam,
                            ev_threshold=args.ev, bankroll=args.bankroll)
    else:
        ap.error("--backtest veya --date YYYY-MM-DD belirtin.")


if __name__ == "__main__":
    main()
