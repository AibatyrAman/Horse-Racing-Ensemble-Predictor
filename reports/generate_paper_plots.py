import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Akademik/Makale stili için ayarlar
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.autolayout': True
})

def format_percentage(x, pos):
    return f"{x:.0%}"

# Veriyi oku
csv_path = "/Users/aibatyr/Documents/Ganyan/reports/model_comparison.csv"
output_dir = "/Users/aibatyr/Documents/Ganyan/reports"
df = pd.read_csv(csv_path)

# Hedef değişkenlere göre ayır
df_winner = df[df["Target"] == "Is_Winner"].sort_values("AUC_mean", ascending=False).copy()
df_top3 = df[df["Target"] == "Is_Top3"].sort_values("AUC_mean", ascending=False).copy()

def plot_target_metrics(data, target_name, file_suffix, show_roi=True):
    # ROI yalnızca Is_Winner için anlamlı (veride sadece kazanma ganyanı var).
    # Top3'te ROI sütunu yok → 2 panel; Is_Winner'da 3 panel.
    has_roi = show_roi and "ROI_value" in data.columns and data["ROI_value"].notna().any()
    n_panels = 3 if has_roi else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    fig.suptitle(f"Model Comparison - Target: {target_name}", fontsize=18, fontweight='bold', y=1.05)

    # 1. AUC Plot
    sns.barplot(data=data, x="AUC_mean", y="Model", ax=axes[0], color="steelblue", edgecolor="black")
    axes[0].set_title("AUC Score (Test Performance)")
    axes[0].set_xlabel("AUC Mean")
    axes[0].set_ylabel("Algorithm")
    axes[0].set_xlim(data["AUC_mean"].min() - 0.02, data["AUC_mean"].max() + 0.01)

    for i, v in enumerate(data["AUC_mean"]):
        axes[0].text(v + 0.001, i, f"{v:.4f}", color='black', va='center')

    # 2. Precision@1 Plot (Win/Hit Rate)
    sns.barplot(data=data.sort_values("P@1", ascending=False), x="P@1", y="Model", ax=axes[1], color="mediumseagreen", edgecolor="black")
    axes[1].set_title("Precision@1 (Top Pick Accuracy)")
    axes[1].set_xlabel("Accuracy Rate")
    axes[1].set_ylabel("")
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(format_percentage))

    for i, v in enumerate(data.sort_values("P@1", ascending=False)["P@1"]):
        axes[1].text(v + 0.005, i, f"{v:.1%}", color='black', va='center')

    # 3. ROI Value Strategy Plot (yalnız Is_Winner)
    if has_roi:
        data_roi = data.dropna(subset=["ROI_value"]).sort_values("ROI_value", ascending=False)
        colors = ['crimson' if val < 0 else 'gold' for val in data_roi["ROI_value"]]
        sns.barplot(data=data_roi, x="ROI_value", y="Model", ax=axes[2], palette=colors, edgecolor="black")
        axes[2].set_title("Value Strategy ROI (Return on Investment)")
        axes[2].set_xlabel("ROI")
        axes[2].set_ylabel("")
        axes[2].xaxis.set_major_formatter(plt.FuncFormatter(format_percentage))

        for i, v in enumerate(data_roi["ROI_value"]):
            axes[2].text(v + (0.01 if v > 0 else -0.05), i, f"{v:+.1%}", color='black', va='center')

        axes[2].axvline(x=-0.262, color='r', linestyle='--', label="Public Baseline (-26.2%)")
        axes[2].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"academic_plot_{file_suffix}.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Grafikleri çizdir
plot_target_metrics(df_winner, "Is_Winner (Predicting 1st Place)", "is_winner", show_roi=True)
plot_target_metrics(df_top3, "Is_Top3 (Predicting Show / Top 3)", "is_top3", show_roi=False)

# Markdown formatında makale tablosu üret
# Tablo 1 (Is_Winner): ROI dahil — kazanma bahsi için anlamlı
md_table = """
### Tablo 1: Kazanan Tahmini İçin Model Performansları (Is_Winner)
| Model | AUC | Precision@1 | Value Strategy ROI |
|-------|-----|-------------|--------------------|
"""
for _, row in df_winner.iterrows():
    roi_txt = f"{row['ROI_value']:+.1%}" if pd.notna(row.get('ROI_value')) else "—"
    md_table += f"| {row['Model']} | {row['AUC_mean']:.4f} | {row['P@1']:.1%} | {roi_txt} |\n"

# Tablo 2 (Is_Top3): ROI YOK (veride plase ganyanı yok) → Precision@3 raporlanır
md_table += """
### Tablo 2: Tabela Tahmini İçin Model Performansları (Is_Top3)

> Not: Top3 için ROI raporlanmaz — veride yalnızca kazanma ganyanı bulunduğundan
> plase finişe parasal getiri hesaplamak yanıltıcı olur. Değerlendirme sıralama
> metrikleriyle yapılır.

| Model | AUC | Precision@1 | Precision@3 |
|-------|-----|-------------|-------------|
"""
for _, row in df_top3.iterrows():
    md_table += f"| {row['Model']} | {row['AUC_mean']:.4f} | {row['P@1']:.1%} | {row['P@3']:.1%} |\n"

with open(os.path.join(output_dir, "academic_tables.md"), "w", encoding="utf-8") as f:
    f.write(md_table)

print("\n✓ Akademik grafikler ve tablolar üretildi. Dizin: /Users/aibatyr/Documents/Ganyan/reports/")
