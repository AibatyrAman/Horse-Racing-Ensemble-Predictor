#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  PİYASA SİNYALİ ABLASYONU — Karşılaştırma Üretici
================================================================================
  Tam model (Ganyan dahil) ile Ablation model (Ganyansız) sonuçlarını yan yana
  koyar. Girdi:
      reports/model_comparison.csv            (tam model)
      reports/model_comparison_ablation.csv   (Ganyansız)
  Çıktı:
      reports/ablation_comparison.md          (hedef başına ΔAUC / ΔP@1 tablosu)
      reports/ablation_auc_is_winner.png      (gruplu bar: tam vs Ganyansız)
      reports/ablation_auc_is_top3.png
================================================================================
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
FULL_CSV   = os.path.join(BASE_DIR, "model_comparison.csv")
ABL_CSV    = os.path.join(BASE_DIR, "model_comparison_ablation.csv")
OUT_MD     = os.path.join(BASE_DIR, "ablation_comparison.md")


def load_and_merge():
    full = pd.read_csv(FULL_CSV)
    abl  = pd.read_csv(ABL_CSV)

    keep = ["Target", "Model", "AUC_mean", "P@1", "P@3"]
    full = full[keep].rename(columns={"AUC_mean": "AUC_full", "P@1": "P1_full", "P@3": "P3_full"})
    abl  = abl[keep].rename(columns={"AUC_mean": "AUC_abl", "P@1": "P1_abl", "P@3": "P3_abl"})

    m = full.merge(abl, on=["Target", "Model"], how="inner")
    m["dAUC"] = m["AUC_full"] - m["AUC_abl"]
    m["dP1"]  = m["P1_full"]  - m["P1_abl"]
    return m


def write_markdown(merged):
    lines = []
    lines.append("# Piyasa Sinyali Ablasyonu — Tam Model vs Ganyansız Model\n")
    lines.append(
        "> `Ganyan_*` (bahis oranı) özellikleri çıkarılarak modelin SADECE projenin kendi\n"
        "> verileriyle (handikap, jokey/antrenör, soy hattı, idman) ne kadar başarılı olduğu\n"
        "> ölçülmüştür. ΔAUC küçükse → projenin kendi feature'ları tek başına güçlüdür.\n"
    )

    target_titles = {
        "Is_Winner": "Tablo A: Kazanan Tahmini (Is_Winner)",
        "Is_Top3":   "Tablo B: Tabela Tahmini (Is_Top3)",
    }

    for target in ["Is_Winner", "Is_Top3"]:
        sub = merged[merged["Target"] == target].sort_values("AUC_full", ascending=False)
        if sub.empty:
            continue
        lines.append(f"\n### {target_titles.get(target, target)}\n")
        lines.append("| Model | AUC (tam) | AUC (Ganyansız) | ΔAUC | P@1 (tam) | P@1 (Ganyansız) | ΔP@1 |")
        lines.append("|-------|-----------|------------------|------|-----------|------------------|------|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['Model']} | {r['AUC_full']:.4f} | {r['AUC_abl']:.4f} | "
                f"{r['dAUC']:+.4f} | {r['P1_full']:.1%} | {r['P1_abl']:.1%} | {r['dP1']:+.1%} |"
            )
        mean_dauc = sub["dAUC"].mean()
        lines.append(f"\n*Ortalama ΔAUC ({target}): {mean_dauc:+.4f} "
                     f"(piyasa sinyalinin ortalama katkısı).*")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  ✓ Markdown kaydedildi: {OUT_MD}")


def plot_target(merged, target, file_suffix):
    sub = merged[merged["Target"] == target].sort_values("AUC_full", ascending=False)
    if sub.empty:
        return
    long = sub.melt(id_vars="Model", value_vars=["AUC_full", "AUC_abl"],
                    var_name="Senaryo", value_name="AUC")
    long["Senaryo"] = long["Senaryo"].map({"AUC_full": "Tam (Ganyan dahil)",
                                            "AUC_abl": "Ganyansız (ablation)"})

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=long, y="Model", x="AUC", hue="Senaryo", ax=ax, edgecolor="black")
    ax.set_title(f"Piyasa Sinyali Ablasyonu — {target}\n(Tam model vs Ganyansız)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("AUC")
    ax.set_ylabel("")
    ax.set_xlim(max(0.5, long["AUC"].min() - 0.03), long["AUC"].max() + 0.01)
    ax.legend(loc="lower right", title="")
    plt.tight_layout()
    out = os.path.join(BASE_DIR, f"ablation_auc_{file_suffix}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Grafik kaydedildi: {out}")


def main():
    if not os.path.isfile(ABL_CSV):
        raise SystemExit(
            f"[HATA] {ABL_CSV} yok. Önce şunu çalıştır:\n"
            f"       python tjk_stage4_modeling.py --ablation"
        )
    merged = load_and_merge()
    write_markdown(merged)
    plot_target(merged, "Is_Winner", "is_winner")
    plot_target(merged, "Is_Top3", "is_top3")
    print("\n✓ Ablation karşılaştırması üretildi.")


if __name__ == "__main__":
    main()
