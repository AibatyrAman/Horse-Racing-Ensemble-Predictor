#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  OLASILIK KALİBRASYONU — Üretici (sızıntısız OOF üzerinde)
================================================================================
  Modelin ürettiği olasılıklar gerçekçi mi? "Model %30 diyorsa gerçekten ~%30 mü
  kazanıyor?" Reliability diagram + Brier skoru + ECE ile ölçer.

  Önemli: Production modeller sınıf-dengeleme (is_unbalance / auto_class_weights /
  scale_pos_weight) kullanır → olasılıklar SIRALAMA (AUC, P@1) için iyi ama
  kalibrasyon için bozuk (genelde over-confident) olabilir. Bu rapor bunu ÖLÇER;
  düzeltme (Platt/isotonic) yapmaz — backtest/forward ROI uçurumunu yorumlamaya
  yardımcı olur.

  Girdi:  data/oof_predictions.csv  (Stage 4 --dump-oof)
  Çıktı:  reports/calibration_is_winner.png
          reports/calibration_is_top3.png
          reports/calibration_summary.md
================================================================================
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # reports/
ROOT     = os.path.dirname(BASE_DIR)                           # proje kökü
OOF_CSV  = os.path.join(ROOT, "data", "oof_predictions.csv")
OUT_MD   = os.path.join(BASE_DIR, "calibration_summary.md")

# (hedef etiketi, gerçek kolon, ham olasılık, kalibre olasılık, dosya eki, başlık)
TARGETS = [
    ("Is_Winner", "Is_Winner", "oof_prob_winner", "oof_prob_winner_cal", "is_winner", "Kazanan (Is_Winner)"),
    ("Is_Top3",   "Is_Top3",   "oof_prob_top3",   "oof_prob_top3_cal",   "is_top3",   "İlk-3 (Is_Top3)"),
]
N_BINS = 10


def reliability(y, p, n_bins=N_BINS):
    """Eşit-genişlik binlerde (ort_tahmin, gözlenen_oran, n) + Brier + ECE."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    brier = float(np.mean((p - y) ** 2))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows, ece, N = [], 0.0, len(p)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        nb = int(mask.sum())
        if nb == 0:
            rows.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": 0,
                         "ort_tahmin": np.nan, "gozlenen": np.nan})
            continue
        mp, mo = float(p[mask].mean()), float(y[mask].mean())
        ece += (nb / N) * abs(mp - mo)
        rows.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": nb,
                     "ort_tahmin": mp, "gozlenen": mo})
    return pd.DataFrame(rows), brier, float(ece)


def plot_reliability(tbl, title, base_rate, out_png, tbl_cal=None):
    valid = tbl.dropna(subset=["ort_tahmin"])
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Mükemmel kalibrasyon")
    ax.plot(valid["ort_tahmin"], valid["gozlenen"], "o-", color="#1f77b4", lw=1.8,
            label="Model (ham)")
    # Bin büyüklüğünü işaret boyutuyla göster
    sizes = (valid["n"] / valid["n"].max() * 250).clip(lower=20)
    ax.scatter(valid["ort_tahmin"], valid["gozlenen"], s=sizes, color="#1f77b4",
               alpha=0.35, zorder=3)
    if tbl_cal is not None:
        vc = tbl_cal.dropna(subset=["ort_tahmin"])
        ax.plot(vc["ort_tahmin"], vc["gozlenen"], "s-", color="#2ca02c", lw=1.8,
                label="Model (isotonic kalibre)")
    ax.axhline(base_rate, color="#d62728", ls=":", lw=1,
               label=f"Taban oran ({base_rate:.1%})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Ortalama tahmin edilen olasılık")
    ax.set_ylabel("Gözlenen frekans (gerçek)")
    ax.set_title(f"Reliability Diagram — {title}")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    if not os.path.isfile(OOF_CSV):
        raise SystemExit(f"[HATA] {OOF_CSV} yok. Önce: python src/tjk_stage4_modeling.py --dump-oof")
    df = pd.read_csv(OOF_CSV, encoding="utf-8-sig")

    lines = ["# Olasılık Kalibrasyonu — OOF (sızıntısız)\n",
             "> Model %X diyorsa gerçekten ~%X mı oluyor? **Brier** (düşük=iyi), **ECE** "
             "(beklenen kalibrasyon hatası, düşük=iyi) ve reliability diagram ile ölçülür. "
             "Production modeller sınıf-dengeleme kullandığından olasılıklar sıralama için iyi "
             "ama kalibrasyon için bozuk (genelde **over-confident**) olabilir — bu, backtest'in "
             "neden forward-test'ten iyimser çıktığını kısmen açıklar.\n",
             f"**Kapsam:** {len(df):,} at-koşu kaydı (OOF).\n"]

    for tname, ycol, pcol, pcol_cal, suffix, title in TARGETS:
        sub = df.dropna(subset=[ycol, pcol])
        y, p = sub[ycol].to_numpy(), sub[pcol].to_numpy()
        tbl, brier, ece = reliability(y, p)
        base = float(np.mean(y))

        tbl_cal = brier_cal = ece_cal = None
        if pcol_cal in df.columns:
            sub_c = df.dropna(subset=[ycol, pcol_cal])
            if len(sub_c):
                tbl_cal, brier_cal, ece_cal = reliability(
                    sub_c[ycol].to_numpy(), sub_c[pcol_cal].to_numpy())

        out_png = os.path.join(BASE_DIR, f"calibration_{suffix}.png")
        plot_reliability(tbl, title, base, out_png, tbl_cal=tbl_cal)

        # Over/under-confidence: yüksek-tahmin binlerinde tahmin > gözlenen mi?
        hi = tbl.dropna(subset=["ort_tahmin"])
        hi = hi[hi["ort_tahmin"] >= 0.5]
        verdict = "—"
        if len(hi):
            diff = float((hi["ort_tahmin"] - hi["gozlenen"]).mean())
            verdict = ("over-confident (tahmin > gerçek)" if diff > 0.03 else
                       "under-confident (tahmin < gerçek)" if diff < -0.03 else
                       "yaklaşık kalibre")

        lines.append(f"\n## {title}\n")
        lines.append(f"- **Brier (ham):** {brier:.4f}  •  **ECE (ham):** {ece:.4f}  •  "
                     f"taban oran: {base:.1%}")
        if ece_cal is not None:
            lines.append(f"- **Brier (isotonic):** {brier_cal:.4f}  •  "
                         f"**ECE (isotonic):** {ece_cal:.4f}")
        lines.append(f"- Yüksek-olasılık bölgesi (≥0.5, ham): **{verdict}**")
        lines.append(f"- Grafik: `reports/calibration_{suffix}.png`\n")
        lines.append("| Bin | n | Ort. tahmin | Gözlenen |")
        lines.append("|-----|---|-------------|----------|")
        for _, r in tbl.iterrows():
            if r["n"] == 0:
                lines.append(f"| {r['bin']} | 0 | — | — |")
            else:
                lines.append(f"| {r['bin']} | {int(r['n'])} | "
                             f"{r['ort_tahmin']:.3f} | {r['gozlenen']:.3f} |")
        cal_note = f" | isotonic ECE={ece_cal:.4f}" if ece_cal is not None else ""
        print(f"  {title}: Brier={brier:.4f} ECE={ece:.4f} taban={base:.1%} → {verdict}{cal_note}")

    lines.append("\n---\n> **Yorum:** Üretim zinciri (Stage 6 tahmin + Stage 8 EV) olasılıkları "
                 "**isotonic** kalibratörden geçirir; yukarıdaki 'isotonic' satırı bu düzeltmenin "
                 "OOF üzerindeki etkisidir. Isotonic monotondur → sıralama metrikleri (AUC, P@1) "
                 "değişmez, yalnız EV gerçekçileşir.")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  ✅ Kalibrasyon raporu: {OUT_MD}")
    print(f"  → reports/calibration_is_winner.png, reports/calibration_is_top3.png")


if __name__ == "__main__":
    main()
