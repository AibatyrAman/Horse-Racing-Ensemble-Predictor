"""
Ganyancı iddiaları — hızlı istatistiksel test (model beklemeden, ham veriden).

İddia 1 (cinsiyet): "Erkek ağırlıklı bir yarışta azınlıkta kalan dişi atın
kazanma şansı yüksektir." → Azınlık-dişi (yarışta dişi oranı ≤ %25) atların
kazanma oranı, şans beklentisi (Σ 1/yarış_at_sayısı) ve erkek rakiplerinin
oranıyla karşılaştırılır; binom testi ile anlamlılık ölçülür.

İddia 2 (don/renk): bedava yan bakış — don bazında kazanma oranları.

Kaynak: data/yaris_ana_tablo.csv (tüm tarihsel sonuçlar).
Çıktı : reports/ganyanci_iddialari.md
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, os.path.join(ROOT, "src"))

from tjk_stage3_feature_engineering import (  # noqa: E402
    parse_cinsiyet, parse_don, extract_at_id_from_url, normalize_sehir,
    CINSIYET_DISI, CINSIYET_ERKEK,
)

YARIS_CSV = os.path.join(ROOT, "data", "yaris_ana_tablo.csv")
OUT_MD = os.path.join(BASE_DIR, "ganyanci_iddialari.md")

CINSIYET_AD = {"d": "dişi tay", "k": "kısrak", "e": "erkek tay",
               "a": "aygır", "g": "iğdiş"}
DON_AD = {"d": "doru", "a": "al", "k": "kır", "y": "yağız"}


def main():
    y = pd.read_csv(YARIS_CSV, encoding="utf-8-sig")
    y["Tarih_dt"] = pd.to_datetime(y["Tarih"], format="%d.%m.%Y", errors="coerce")
    y["Siralama"] = pd.to_numeric(y["Siralama"], errors="coerce")
    y["at_id"] = y["At_URL"].apply(extract_at_id_from_url)
    y = y.dropna(subset=["at_id", "Tarih_dt"]).copy()
    y["Is_Winner"] = (y["Siralama"] == 1).astype(int)
    y["Is_Top3"] = (y["Siralama"] <= 3).astype(int)
    y["Unique_Race_ID"] = (
        y["Tarih_dt"].dt.strftime("%Y%m%d") + "_" +
        y["Sehir"].apply(normalize_sehir) + "_" + y["Kosu_ID"].astype(str)
    )

    y["Cinsiyet"] = y["Yas"].apply(parse_cinsiyet)
    y["Don"] = y["Yas"].apply(parse_don)
    y["Disi"] = np.where(y["Cinsiyet"].isin(list(CINSIYET_DISI)), 1.0,
                         np.where(y["Cinsiyet"].isin(list(CINSIYET_ERKEK)), 0.0, np.nan))

    parsed = y["Cinsiyet"].notna().mean()
    print(f"  → {len(y):,} satır | cinsiyet parse oranı: %{parsed*100:.1f}")

    lines = ["# Ganyancı İddiaları — İstatistiksel Test\n",
             f"**Kaynak:** `yaris_ana_tablo.csv` ({len(y):,} at-koşu kaydı, "
             f"{y['Unique_Race_ID'].nunique():,} yarış). "
             f"Cinsiyet parse oranı %{parsed*100:.1f}.\n"]

    # ── Genel cinsiyet kırılımı ─────────────────────────────────────────
    lines.append("## 0) Genel: cinsiyet bazında kazanma oranları\n")
    lines.append("| Cinsiyet | n | Kazanma | Tabela |")
    lines.append("|----------|---|---------|--------|")
    for kod, grp in y.groupby("Cinsiyet"):
        lines.append(f"| {CINSIYET_AD.get(kod, kod)} ({kod}) | {len(grp):,} "
                     f"| %{grp['Is_Winner'].mean()*100:.1f} "
                     f"| %{grp['Is_Top3'].mean()*100:.1f} |")

    # ── İddia 1: Azınlık dişi ──────────────────────────────────────────
    race = y.dropna(subset=["Disi"]).copy()
    race["Yaris_At_Sayisi"] = race.groupby("Unique_Race_ID")["at_id"].transform("count")
    race["Disi_Orani"] = race.groupby("Unique_Race_ID")["Disi"].transform("mean")

    lines.append("\n## 1) İddia: erkek ağırlıklı yarışta azınlık dişi avantajlı mı?\n")
    lines.append("Azınlık dişi = yarıştaki dişi oranı ≤ eşik VE at dişi. Şans beklentisi = "
                 "Σ(1/yarış_at_sayısı)/n — her atın rastgele kazanma olasılığı.\n")
    lines.append("| Eşik | n (azınlık dişi) | Kazanma | Şans beklentisi | Aynı yarış erkekleri | Binom p |")
    lines.append("|------|------------------|---------|-----------------|----------------------|---------|")

    for esik in [0.15, 0.25, 0.35]:
        az = race[(race["Disi"] == 1.0) & (race["Disi_Orani"] <= esik)]
        if len(az) == 0:
            lines.append(f"| ≤{esik:.0%} | 0 | — | — | — | — |")
            continue
        n = len(az)
        wins = int(az["Is_Winner"].sum())
        exp_p = float((1.0 / az["Yaris_At_Sayisi"]).mean())
        # Aynı yarışlardaki erkeklerin kazanma oranı (adil kıyas)
        erkek_ayni = race[(race["Disi"] == 0.0) &
                          (race["Unique_Race_ID"].isin(az["Unique_Race_ID"]))]
        erkek_rate = erkek_ayni["Is_Winner"].mean() if len(erkek_ayni) else float("nan")
        p = binomtest(wins, n, exp_p, alternative="two-sided").pvalue
        lines.append(f"| ≤{esik:.0%} | {n:,} | %{wins/n*100:.1f} | %{exp_p*100:.1f} "
                     f"| %{erkek_rate*100:.1f} | {p:.3g} |")

    # Karma yarışlarda genel dişi-vs-erkek (kompozisyondan bağımsız)
    karma = race[(race["Disi_Orani"] > 0) & (race["Disi_Orani"] < 1)]
    d = karma[karma["Disi"] == 1.0]
    e = karma[karma["Disi"] == 0.0]
    lines.append(f"\nKarma yarışlarda genel: dişi %{d['Is_Winner'].mean()*100:.1f} "
                 f"(n={len(d):,}) vs erkek %{e['Is_Winner'].mean()*100:.1f} (n={len(e):,}).")

    # ── İddia 2 (bonus): Don ───────────────────────────────────────────
    lines.append("\n## 2) Bonus: don (tüy rengi) bazında kazanma oranı\n")
    lines.append("| Don | n | Kazanma | Tabela |")
    lines.append("|-----|---|---------|--------|")
    for kod, grp in y.dropna(subset=["Don"]).groupby("Don"):
        if len(grp) < 50:
            continue
        lines.append(f"| {DON_AD.get(kod, kod)} ({kod}) | {len(grp):,} "
                     f"| %{grp['Is_Winner'].mean()*100:.1f} "
                     f"| %{grp['Is_Top3'].mean()*100:.1f} |")
    lines.append("\n> Don biyolojik bir performans belirleyici değildir; farklar büyük "
                 "olasılıkla soy hattı karışıklığıdır (confounding). Modelde kullanılmaz, "
                 "yalnız merak gideren yan bakış.")

    lines.append("\n---\n*Bu rapor tek-değişken (univariate) bakıştır; nihai hüküm, "
                 "feature'ın çok-değişkenli modeldeki katkısıyla (ablation ΔAUC) verilir.*")

    os.makedirs(BASE_DIR, exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  ✅ Rapor: {OUT_MD}")
    print("\n".join(lines[6:]))


if __name__ == "__main__":
    main()
