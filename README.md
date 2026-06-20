# Predictive Racing Intelligence (TJK) 🏇

Türkiye Jokey Kulübü (TJK) at yarışı sonuçlarını tahmin eden, uçtan uca makine öğrenmesi
pipeline'ı. Veri kazımadan (scraping) özellik mühendisliğine, model eğitiminden ROI
değerlendirmesine kadar dört aşamadan oluşur. Akademik bir tez/makale çalışmasıdır.

> **Sıfır veri sızıntısı** ilkesiyle tasarlanmıştır: ileri-zincirli zaman-bazlı CV
> (TimeSeriesSplit), fold-içi imputation ve sızıntısız target encoding kullanılır.
> Metodoloji ve düzeltmelerin ayrıntısı için bkz. **[RAPOR.md](RAPOR.md)**.

---

## Pipeline Aşamaları

| Aşama | Dosya | İşlev | Çıktı |
|-------|-------|-------|-------|
| 1 | `tjk_scraper_stage1.py` | Yarış sonuçları scraping (Selenium) | `yaris_ana_tablo.csv` |
| 2 | `tjk_scraper_stage2.py` | At profili + idman verileri | `atlar_statik_tablo.csv`, `idmanlar_tablo.csv` |
| 3 | `tjk_stage3_feature_engineering.py` | Veri birleştirme + feature engineering | `master_feature_matrix.csv` |
| 4 | `tjk_stage4_modeling.py` | Model eğitimi + ensemble + ROI | `models/`, `reports/` |

**Hedef değişkenler:** `Is_Winner` (1. olma), `Is_Top3` (ilk 3'e girme).
**Veri:** ~21.6k at-yarış kaydı, 2.199 yarış, 127 yarış günü (2025-04 → 2026-03).

---

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install pandas numpy scikit-learn xgboost lightgbm catboost \
            optuna shap matplotlib seaborn beautifulsoup4 selenium tqdm
```

Scraping aşamaları (1–2) Google Chrome + uyumlu ChromeDriver gerektirir.

---

## Kullanım

```bash
# Tüm pipeline (Stage 1 → 4)
python tjk_pipeline.py

# Belirli aşamadan başla / tek aşama
python tjk_pipeline.py --from 3        # Stage 3 + 4
python tjk_pipeline.py --only 4        # Sadece modelleme

# Modelleme — tam model
python tjk_stage4_modeling.py

# Modelleme — piyasa sinyali ablasyonu (Ganyan_* olmadan)
python tjk_stage4_modeling.py --ablation

# Raporlar / grafikler
python reports/generate_paper_plots.py
python reports/generate_ablation_comparison.py
```

Scraper'lar **resume** destekler: yarıda durdurulursa (Ctrl+C) tekrar çalıştırıldığında
kaldığı yerden devam eder.

---

## Çıktılar

- `models/` — eğitilmiş modeller (`.pkl`), metadata (`.json`), `production_registry.json`
- `reports/model_comparison.csv` — tüm modellerin karşılaştırma tablosu
- `reports/academic_tables.md`, `reports/academic_plot_*.png` — makale tabloları/grafikleri
- `reports/ablation_comparison.md`, `reports/ablation_auc_*.png` — piyasa ablasyonu
- `reports/fi_*` — feature importance + SHAP grafikleri

---

## Metodoloji Notları

- **Değerlendirme:** İleri-zincirli `TimeSeriesSplit` (geçmişle eğit → gelecekle test).
  Imputation/encoding her fold'da yalnızca eğitim kısmında fit edilir.
- **ROI:** Yalnızca `Is_Winner` için raporlanır (veride sadece kazanma ganyanı var).
  `Is_Top3` sıralama metrikleriyle (P@1, P@3) değerlendirilir.
- **Ablasyon bulgusu:** Tüm piyasa sinyali (`Ganyan_*`) çıkarıldığında AUC yalnızca
  ~0.01–0.02 düşer → modelin gücü büyük ölçüde projenin kendi verilerinden (handikap,
  jokey/antrenör, soy hattı, idman) gelir.

> ⚠️ Bu proje akademik/araştırma amaçlıdır. Gerçek bahis kararları için kullanılması
> önerilmez; ROI sonuçları geçmiş verilere dayalıdır ve yüksek varyans içerir.

---

## Lisans

(Belirtilecek — örn. MIT veya akademik kullanım.)
