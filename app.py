#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  TJK Tahmin Paneli — Streamlit Arayüzü (yerel)
================================================================================
  Çalıştır:  streamlit run app.py
  Mevcut CLI scriptlerini (Stage 5/6/7, pipeline, retrain) subprocess ile sarar
  ve çıktıları (predictions_log, live_performance, model_comparison, reports/*)
  gösterir. Ağır iş YOK — yalnız okuma + buton tetikleme.
================================================================================
"""

import os
import sys
import json
import subprocess
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(BASE_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

PRED_LOG  = os.path.join(BASE_DIR, "predictions_log.csv")
PERF_CSV  = os.path.join(BASE_DIR, "live_performance.csv")
MODEL_CMP = os.path.join(BASE_DIR, "reports", "model_comparison.csv")
ABL_MD    = os.path.join(BASE_DIR, "reports", "ablation_comparison.md")
REG_FULL  = os.path.join(BASE_DIR, "models", "production_registry.json")
REG_ABL   = os.path.join(BASE_DIR, "models", "production_registry_ablation.json")
REPORTS   = os.path.join(BASE_DIR, "reports")
PY = sys.executable

# İşlemler: anahtar → (etiket, komut)
JOBS = {
    "scrape":   ("Programı Çek (bugün)",        [PY, "tjk_stage5_live_program.py", "--headless"]),
    "predict":  ("Tahmin Üret",                 [PY, "tjk_stage6_predict.py"]),
    "strategy": ("Strateji Üret (bugün)",       "STRATEGY"),  # özel: en güncel tarihe
    "results":  ("Sonuç Çek + Değerlendir",     "RESULTS"),  # özel: iki adım
    "backtest": ("Strateji Backtest",           [PY, "tjk_stage8_betting_strategy.py", "--backtest"]),
    "retrain":  ("Yeniden Eğit (uzun sürer)",   [PY, "tjk_retrain_monitor.py", "--run"]),
}

STRAT_SUMMARY = os.path.join(BASE_DIR, "reports", "betting_strategy_summary.md")
BANKROLL_PNG  = os.path.join(BASE_DIR, "reports", "bankroll_curve.png")


# ─────────────────────────────────────────────────────────────────────────────
#  JOB RUNNER (dosya-tabanlı; Streamlit yeniden-çalıştırmasına dayanıklı)
# ─────────────────────────────────────────────────────────────────────────────
def _log_path(key):  return os.path.join(RUNS_DIR, f"{key}.log")
def _done_path(key): return os.path.join(RUNS_DIR, f"{key}.done")


def job_status(key):
    log, done = _log_path(key), _done_path(key)
    if not os.path.exists(log):
        return "idle", None
    if os.path.exists(done):
        try:
            code = int(open(done).read().strip() or "0")
        except Exception:
            code = 0
        return "done", code
    return "running", None


def any_running():
    return any(job_status(k)[0] == "running" for k in JOBS)


def start_job(key):
    cmd = JOBS[key][1]
    log = _log_path(key)
    done = _done_path(key)
    if os.path.exists(done):
        os.remove(done)
    # Komutu bash sarmalayıcıyla çalıştır: bitince dönüş kodunu .done'a yaz
    if cmd == "RESULTS":
        inner = (f'"{PY}" tjk_pipeline.py --only 1 && "{PY}" tjk_stage7_reconcile.py')
    elif cmd == "STRATEGY":
        # En güncel tahmin tarihine göre öneri üret
        d = read_predictions()
        if d is not None and not d.empty:
            latest = d["Tarih"].dropna().dt.date.max().strftime("%Y-%m-%d")
            inner = f'"{PY}" tjk_stage8_betting_strategy.py --date {latest}'
        else:
            inner = 'echo "Önce tahmin üret (predictions_log.csv yok)"'
    else:
        inner = " ".join(f'"{c}"' for c in cmd)
    wrapper = f'{inner} > "{log}" 2>&1; echo $? > "{done}"'
    subprocess.Popen(wrapper, shell=True, cwd=BASE_DIR)


def tail(path, n=45):
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        return "".join(f.readlines()[-n:])


# ─────────────────────────────────────────────────────────────────────────────
#  VERİ OKUYUCULAR
# ─────────────────────────────────────────────────────────────────────────────
def read_predictions():
    if not os.path.isfile(PRED_LOG):
        return None
    df = pd.read_csv(PRED_LOG, encoding="utf-8-sig")
    df["Tarih"] = pd.to_datetime(df["Tarih"], errors="coerce")
    return df


def _kosu_label(g):
    """Koşu etiketi: ID + (varsa) saat — örn. '225686 — 16:45'."""
    kid = str(g["Kosu_ID"].iloc[0])
    if "Kosu_Saati" in g.columns:
        s = g["Kosu_Saati"].dropna().astype(str)
        s = s[s.str.strip().ne("") & s.ne("nan")]
        if not s.empty:
            return f"{kid} — {s.iloc[0]}"
    return kid


def race_summary(df_day):
    """Yarış-yarış özet: full/abl favori + piyasa favorisi."""
    rows = []
    for rid, g in df_day.groupby("Unique_Race_ID"):
        def pick(col):
            if col not in g or g[col].isna().all():
                return ("—", float("nan"))
            r = g.loc[g[col].idxmax()]
            return (r["At_Adi"], float(r[col]))
        af, pf = pick("prob_winner_full")
        aa, pa = pick("prob_winner_abl")
        if g["Ganyan_Sayi"].notna().any():
            fav = g.loc[g["Ganyan_Sayi"].idxmin(), "At_Adi"]
        else:
            fav = "—"
        rows.append({
            "Koşu": _kosu_label(g),
            "Model (full)": af, "P(win)": pf,
            "Model (ganyansız)": aa, "P(win) abl": pa,
            "Piyasa favorisi": fav,
            "Value?": "★" if (af != fav and af != "—") else "",
        })
    return pd.DataFrame(rows)


def race_summary_top3(df_day):
    """Yarış-yarış TABELA (ilk 3) tahmini: model top-3 vs piyasa top-3."""
    rows = []
    for rid, g in df_day.groupby("Unique_Race_ID"):
        def top3(col, ascending=False):
            if col not in g or g[col].isna().all():
                return "—"
            picks = g.sort_values(col, ascending=ascending).head(3)["At_Adi"].astype(str)
            return ", ".join(picks)
        rows.append({
            "Koşu": _kosu_label(g),
            "Tabela — model (full)":      top3("prob_top3_full"),
            "Tabela — model (ganyansız)": top3("prob_top3_abl"),
            "Piyasa ilk 3 (ganyan)":      top3("Ganyan_Sayi", ascending=True),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  SAYFA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="TJK Tahmin Paneli", page_icon="🏇", layout="wide")
st.title("🏇 TJK Tahmin Paneli")
st.caption("Yerel araç • araştırma/kâğıt-üzeri amaçlı • gerçek bahis önerilmez")

tab_today, tab_perf, tab_strat, tab_models, tab_ops = st.tabs(
    ["📊 Bugün / Tahminler", "📈 Performans", "💰 Strateji", "🤖 Modeller", "⚙️ İşlemler"]
)

# ── BUGÜN ────────────────────────────────────────────────────────────────────
with tab_today:
    df = read_predictions()
    if df is None or df.empty:
        st.info("Henüz tahmin yok. **İşlemler** sekmesinden *Programı Çek* → *Tahmin Üret* çalıştır.")
    else:
        dates = sorted(df["Tarih"].dropna().dt.date.unique())
        sel = st.selectbox("Tarih", dates, index=len(dates) - 1,
                           format_func=lambda d: d.strftime("%d.%m.%Y"))
        day = df[df["Tarih"].dt.date == sel]
        cities = sorted(day["Sehir"].dropna().astype(str).unique())
        c1, c2, c3 = st.columns(3)
        c1.metric("Yarış", day["Unique_Race_ID"].nunique())
        c2.metric("At", len(day))
        c3.metric("Hipodrom", len(cities))

        view = st.radio("Bahis türü", ["🥇 Kazanan (Ganyan)", "📋 Tabela (İlk 3)"],
                        horizontal=True, label_visibility="collapsed")
        is_winner_view = view.startswith("🥇")

        for city in cities:
            sub = day[day["Sehir"].astype(str) == city]
            with st.expander(f"🏟️ {city}  ({sub['Unique_Race_ID'].nunique()} koşu)", expanded=True):
                if is_winner_view:
                    summ = race_summary(sub)
                    st.dataframe(
                        summ.style.format({"P(win)": "{:.1%}", "P(win) abl": "{:.1%}"}),
                        width="stretch", hide_index=True,
                    )
                else:
                    st.dataframe(race_summary_top3(sub), width="stretch", hide_index=True)

        if is_winner_view:
            st.caption("★ = modelin (full) seçimi piyasa favorisinden farklı (potansiyel *value*).")
        else:
            st.caption("Tabela = ilk 3'e girmesi en olası 3 at. `full` ganyanlı, `ganyansız` "
                       "erken model; piyasa sütunu en düşük 3 ganyan.")

# ── PERFORMANS ───────────────────────────────────────────────────────────────
with tab_perf:
    if not os.path.isfile(PERF_CSV):
        st.info("Henüz canlı performans yok. Yarışlar oynanınca **İşlemler → Sonuç Çek + "
                "Değerlendir** çalıştır.")
    else:
        perf = pd.read_csv(PERF_CSV, encoding="utf-8-sig")
        cum = perf[perf["scope"] == "kümülatif"]
        st.subheader("Kümülatif (forward-test)")
        cols = st.columns(len(cum) if len(cum) else 1)
        for col, (_, r) in zip(cols, cum.iterrows()):
            roi = f"{r['ROI_winner_top1']:+.1%}" if pd.notna(r["ROI_winner_top1"]) else "—"
            col.metric(f"{r['variant']} • P@1 (winner)", f"{r['P@1_winner']:.1%}",
                       help=f"ROI(winner top1)={roi} • {int(r['n_races'])} yarış")
            col.metric(f"{r['variant']} • ROI (winner)", roi)
        st.dataframe(perf, width="stretch", hide_index=True)

        daily = perf[perf["scope"] == "günlük"]
        if daily["Tarih"].nunique() > 1:
            st.subheader("Günlük P@1 (winner) trendi")
            piv = daily.pivot_table(index="Tarih", columns="variant", values="P@1_winner")
            st.line_chart(piv)

# ── STRATEJİ ─────────────────────────────────────────────────────────────────
with tab_strat:
    st.warning("⚠️ Araştırma/kâğıt-üzeri. Egzotik ödemeler **piyasa-ima tahmini** "
               "(gerçek geçmiş ödeme verisi yok). Yüksek varyans → ROI'yi geniş güven "
               "aralığıyla yorumlayın. Gerçek bahis önerilmez.")

    st.subheader("Bugünün önerileri")
    df_pred = read_predictions()
    bets_md = None
    if df_pred is not None and not df_pred.empty:
        latest = df_pred["Tarih"].dropna().dt.date.max()
        bets_path = os.path.join(BASE_DIR, f"bets_{latest.strftime('%Y-%m-%d')}.md")
        if os.path.isfile(bets_path):
            bets_md = open(bets_path, encoding="utf-8").read()
    if bets_md:
        st.markdown(bets_md)
    else:
        st.info("Henüz öneri yok. **İşlemler → Strateji Üret (bugün)** çalıştır "
                "(önce *Tahmin Üret* gerekli).")

    st.divider()
    st.subheader("Backtest (geçmiş, sızıntısız OOF)")
    if os.path.isfile(BANKROLL_PNG):
        st.image(BANKROLL_PNG, width="stretch")
    if os.path.isfile(STRAT_SUMMARY):
        st.markdown(open(STRAT_SUMMARY, encoding="utf-8").read())
    else:
        st.info("Henüz backtest yok. **İşlemler → Strateji Backtest** çalıştır "
                "(önce `python tjk_stage4_modeling.py --dump-oof` ile `oof_predictions.csv`).")

# ── MODELLER ─────────────────────────────────────────────────────────────────
with tab_models:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Production (tam)")
        if os.path.isfile(REG_FULL):
            st.json(json.load(open(REG_FULL, encoding="utf-8")))
        else:
            st.info("Yok — `python tjk_stage4_modeling.py` çalıştır.")
    with c2:
        st.subheader("Production (ganyansız/ablation)")
        if os.path.isfile(REG_ABL):
            st.json(json.load(open(REG_ABL, encoding="utf-8")))
        else:
            st.info("Yok — `python tjk_stage4_modeling.py --ablation` çalıştır.")

    if os.path.isfile(MODEL_CMP):
        st.subheader("Model karşılaştırması")
        st.dataframe(pd.read_csv(MODEL_CMP, encoding="utf-8-sig"),
                     width="stretch", hide_index=True)

    st.subheader("Grafikler")
    imgs = [p for p in [
        os.path.join(REPORTS, "academic_plot_is_winner.png"),
        os.path.join(REPORTS, "academic_plot_is_top3.png"),
        os.path.join(REPORTS, "ablation_auc_is_winner.png"),
        os.path.join(REPORTS, "ablation_auc_is_top3.png"),
    ] if os.path.isfile(p)]
    for i in range(0, len(imgs), 2):
        for col, img in zip(st.columns(2), imgs[i:i + 2]):
            col.image(img, width="stretch")
    if os.path.isfile(ABL_MD):
        with st.expander("Piyasa sinyali ablasyonu (tablo)"):
            st.markdown(open(ABL_MD, encoding="utf-8").read())

# ── İŞLEMLER ─────────────────────────────────────────────────────────────────
with tab_ops:
    st.subheader("Manuel işlemler")
    st.caption("Sıra: **Programı Çek → Tahmin Üret** • (yarışlar oynanınca) **Sonuç Çek + "
               "Değerlendir** • (haftada bir) **Yeniden Eğit**")
    running = any_running()
    if running:
        st.warning("Bir işlem çalışıyor — bitmesini bekleyin.")

    cols = st.columns(len(JOBS))
    for col, (key, (label, _)) in zip(cols, JOBS.items()):
        if col.button(label, disabled=running, width="stretch", key=f"btn_{key}"):
            start_job(key)
            st.rerun()

    st.divider()
    if st.button("🔄 Durumu yenile"):
        st.rerun()

    for key, (label, _) in JOBS.items():
        status, code = job_status(key)
        if status == "idle":
            continue
        icon = {"running": "⏳", "done": "✅" if code == 0 else "❌"}[status]
        head = f"{icon} {label} — {status}" + (f" (kod {code})" if status == "done" else "")
        with st.expander(head, expanded=(status == "running")):
            st.code(tail(_log_path(key)) or "(log boş)", language="bash")
