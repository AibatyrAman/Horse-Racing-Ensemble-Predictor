"""
Panel veri katmanı: CSV/MD dosyalarını mtime-önbellekli okur ve API'nin
döndürdüğü JSON yapılarını kurar. Ağır groupby'lar yalnız dosya değişince
yeniden hesaplanır.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

WEB_DIR  = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.dirname(WEB_DIR)
ROOT     = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR  = os.path.join(ROOT, "outputs")
REPORTS  = os.path.join(ROOT, "reports")
MODELS   = os.path.join(ROOT, "models")

sys.path.insert(0, SRC_DIR)
from tjk_stage3_feature_engineering import extract_at_id_from_url, normalize_sehir  # noqa: E402

PRED_LOG    = os.path.join(DATA_DIR, "predictions_log.csv")
PERF_CSV    = os.path.join(DATA_DIR, "live_performance.csv")
PROGRAM_CSV = os.path.join(DATA_DIR, "program_tablo.csv")
BACKTEST_CSV = os.path.join(DATA_DIR, "betting_strategy_backtest.csv")
MODEL_CMP   = os.path.join(REPORTS, "model_comparison.csv")

_cache = {}


def _read_csv(path):
    """mtime değişmediyse önbellekten döndürür."""
    if not os.path.isfile(path):
        return None
    mtime = os.path.getmtime(path)
    hit = _cache.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    df = pd.read_csv(path, encoding="utf-8-sig")
    _cache[path] = (mtime, df)
    return df


def _read_text(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


def _f(x, nd=4):
    """NaN güvenli float (JSON'da null olur)."""
    try:
        v = float(x)
        return None if np.isnan(v) else round(v, nd)
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  TAHMİNLER
# ─────────────────────────────────────────────────────────────────────────────
def _parse_tarih(s):
    """predictions_log Tarih'i ISO (standart) ya da gg.aa.yyyy (eski) olabilir."""
    d = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    return d.fillna(pd.to_datetime(s, format="%d.%m.%Y", errors="coerce"))


def prediction_dates():
    """[{value: ham string (filtre anahtarı), label: gg.aa.yyyy}] — yeni→eski."""
    df = _read_csv(PRED_LOG)
    if df is None or df.empty:
        return []
    raw = df["Tarih"].astype(str)
    parsed = _parse_tarih(raw)
    pairs = (pd.DataFrame({"value": raw, "dt": parsed}).dropna(subset=["dt"])
             .drop_duplicates("value").sort_values("dt", ascending=False))
    return [{"value": v, "label": dt.strftime("%d.%m.%Y")}
            for v, dt in zip(pairs["value"], pairs["dt"])]


def _program_enrichment():
    """program_tablo'dan (Unique_Race_ID, at_id) → siklet/start/antrenör/URL."""
    prog = _read_csv(PROGRAM_CSV)
    if prog is None or prog.empty:
        return None
    pm = prog.copy()
    pm["at_id"] = pm["At_URL"].apply(extract_at_id_from_url)
    pm = pm.dropna(subset=["at_id"])
    pm["at_id"] = pm["at_id"].astype(int)
    pm["Unique_Race_ID"] = (
        pd.to_datetime(pm["Tarih"], format="%d.%m.%Y", errors="coerce").dt.strftime("%Y%m%d")
        + "_" + pm["Sehir"].apply(normalize_sehir) + "_" + pm["Kosu_ID"].astype(str)
    )
    keep = ["Unique_Race_ID", "at_id", "Siklet", "Start", "Antrenor_Adi", "At_URL",
            "Yas", "Pist_Durumu"]
    keep += [c for c in ["Mesafe", "Yaris_Turu"] if c in pm.columns]
    return pm[keep].drop_duplicates(["Unique_Race_ID", "at_id"])


def predictions_payload(date=None):
    df = _read_csv(PRED_LOG)
    if df is None or df.empty:
        return {"dates": [], "date": None, "cities": []}
    dates = prediction_dates()
    date = date or (dates[0]["value"] if dates else None)
    day = df[df["Tarih"].astype(str) == str(date)].copy()
    if day.empty:
        return {"dates": dates, "date": date, "cities": []}

    enrich = _program_enrichment()
    if enrich is not None:
        day = day.merge(enrich, on=["Unique_Race_ID", "at_id"], how="left",
                        suffixes=("", "_prog"))

    cities = []
    for sehir, city_g in day.groupby("Sehir", sort=True):
        races = []
        for rid, g in city_g.groupby("Unique_Race_ID", sort=True):
            g = g.copy()
            runners = []
            fav_id = None
            if g["Ganyan_Sayi"].notna().any():
                fav_id = int(g.loc[g["Ganyan_Sayi"].idxmin(), "at_id"])
            picks = {}
            for var in ["full", "abl", "blend"]:
                col = f"prob_winner_{var}"
                if col in g.columns and g[col].notna().any():
                    picks[var] = int(g.loc[g[col].idxmax(), "at_id"])
            sort_col = next((f"prob_winner_{v}" for v in ["full", "abl", "blend"]
                             if f"prob_winner_{v}" in g.columns and g[f"prob_winner_{v}"].notna().any()),
                            None)
            if sort_col:
                g = g.sort_values(sort_col, ascending=False)
            for _, r in g.iterrows():
                runners.append({
                    "at_id": int(r["at_id"]),
                    "at": r.get("At_Adi"),
                    "jokey": r.get("Jokey_Adi"),
                    "antrenor": r.get("Antrenor_Adi"),
                    "yas": r.get("Yas"),
                    "siklet": r.get("Siklet"),
                    "start": r.get("Start"),
                    "ganyan": _f(r.get("Ganyan_Sayi"), 2),
                    "url": r.get("At_URL") if isinstance(r.get("At_URL"), str) else None,
                    "p_full": _f(r.get("prob_winner_full")),
                    "p_abl": _f(r.get("prob_winner_abl")),
                    "p_blend": _f(r.get("prob_winner_blend")),
                    "p3_full": _f(r.get("prob_top3_full")),
                    "is_fav": fav_id is not None and int(r["at_id"]) == fav_id,
                    "picks": [v for v, aid in picks.items() if aid == int(r["at_id"])],
                })
            saat = None
            if "Kosu_Saati" in g.columns and g["Kosu_Saati"].notna().any():
                saat = str(g["Kosu_Saati"].dropna().iloc[0])
            races.append({
                "race_id": rid,
                "kosu_id": str(g["Kosu_ID"].iloc[0]),
                "saat": saat,
                "mesafe": _f(g["Mesafe"].dropna().iloc[0], 0) if "Mesafe" in g.columns and g["Mesafe"].notna().any() else None,
                "yaris_turu": (str(g["Yaris_Turu"].dropna().iloc[0])
                               if "Yaris_Turu" in g.columns and g["Yaris_Turu"].notna().any() else None),
                "odds_ts": (str(g["Odds_TS"].dropna().iloc[0])
                            if "Odds_TS" in g.columns and g["Odds_TS"].notna().any() else None),
                "n": len(runners),
                "runners": runners,
                "surpriz": bool(picks.get("full") and fav_id and picks["full"] != fav_id),
            })
            races.sort(key=lambda r: (r["saat"] or "99:99"))
        cities.append({"name": normalize_sehir(sehir) or str(sehir), "races": races})
    return {"dates": dates, "date": date, "cities": cities}


# ─────────────────────────────────────────────────────────────────────────────
#  PERFORMANS
# ─────────────────────────────────────────────────────────────────────────────
def performance_payload():
    df = _read_csv(PERF_CSV)
    if df is None or df.empty:
        return {"cumulative": [], "daily": []}

    def rows(scope):
        sub = df[df["scope"] == scope]
        out = []
        for _, r in sub.iterrows():
            out.append({
                "tarih": str(r["Tarih"]),
                "variant": r["variant"],
                "n_races": int(r["n_races"]),
                "p1_winner": _f(r.get("P@1_winner")),
                "p1_top3": _f(r.get("P@1_top3")),
                "p3_top3": _f(r.get("P@3_top3")),
                "roi": _f(r.get("ROI_winner_top1")),
                "n_bets": _f(r.get("n_bets"), 0),
            })
        return out

    return {"cumulative": rows("kümülatif"), "daily": rows("günlük")}


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEJİ
# ─────────────────────────────────────────────────────────────────────────────
def strategy_payload(bet_type=None):
    # Günün öneri markdown'u
    bets_files = sorted(glob.glob(os.path.join(OUT_DIR, "bets_*.md")), reverse=True)
    bets_md = _read_text(bets_files[0]) if bets_files else None
    bets_name = os.path.basename(bets_files[0]) if bets_files else None

    summary_md = _read_text(os.path.join(REPORTS, "betting_strategy_summary.md"))

    bt = _read_csv(BACKTEST_CSV)
    bet_types, rows = [], []
    if bt is not None and not bt.empty:
        bet_types = sorted(bt["bet_type"].dropna().unique().tolist())
        sub = bt if not bet_type else bt[bt["bet_type"] == bet_type]
        sub = sub.sort_values("Tarih", ascending=False).head(300)
        for _, r in sub.iterrows():
            rows.append({
                "tarih": str(r.get("Tarih", ""))[:10],
                "bet_type": r.get("bet_type"),
                "combo": r.get("horses", r.get("combo")),
                "p_model": _f(r.get("p_model")),
                "p_market": _f(r.get("p_market")),
                "ev": _f(r.get("ev_ratio")),
                "hit": bool(r.get("hit")) if pd.notna(r.get("hit")) else None,
            })
    return {"bets_md": bets_md, "bets_name": bets_name, "summary_md": summary_md,
            "bet_types": bet_types, "backtest": rows,
            "bankroll_png": os.path.isfile(os.path.join(REPORTS, "bankroll_curve.png"))}


# ─────────────────────────────────────────────────────────────────────────────
#  MODELLER
# ─────────────────────────────────────────────────────────────────────────────
def models_payload():
    def _reg(name):
        p = os.path.join(MODELS, name)
        if not os.path.isfile(p):
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    cmp_rows = []
    cmp_df = _read_csv(MODEL_CMP)
    if cmp_df is not None:
        for _, r in cmp_df.iterrows():
            cmp_rows.append({
                "target": r["Target"], "model": r["Model"],
                "auc": _f(r["AUC_mean"]), "p1": _f(r["P@1"]), "p3": _f(r["P@3"]),
                "roi_value": _f(r.get("ROI_value")),
            })

    img_names = ["academic_plot_is_winner.png", "academic_plot_is_top3.png",
                 "ablation_auc_is_winner.png", "ablation_auc_is_top3.png",
                 "calibration_is_winner.png", "calibration_is_top3.png",
                 "fi_LightGBM_Is_Winner_shap.png", "fi_LightGBM_Is_Top3_shap.png"]
    images = [n for n in img_names if os.path.isfile(os.path.join(REPORTS, n))]

    return {"registry_full": _reg("production_registry.json"),
            "registry_abl": _reg("production_registry_ablation.json"),
            "comparison": cmp_rows, "images": images}
