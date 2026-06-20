# Predictive Racing Intelligence (TJK) 🏇

Türkiye Jokey Kulübü (TJK) at yarışı sonuçlarını tahmin eden, uçtan uca makine öğrenmesi
pipeline'ı. Veri kazımadan (scraping) özellik mühendisliğine, model eğitiminden ROI
değerlendirmesine kadar dört aşamadan oluşur. Akademik bir tez/makale çalışmasıdır.

> **Sıfır veri sızıntısı** ilkesiyle tasarlanmıştır: ileri-zincirli zaman-bazlı CV
> (TimeSeriesSplit), fold-içi imputation ve sızıntısız target encoding kullanılır.
> Metodoloji ve düzeltmelerin ayrıntısı için bkz. **[docs/RAPOR.md](docs/RAPOR.md)**.

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

## Dizin Yapısı

```
Ganyan/
├── README.md  •  requirements.txt  •  .gitignore
├── src/                  # TÜM kod
│   ├── app.py            #   Streamlit arayüzü
│   ├── tjk_pipeline.py   #   orkestratör (stage 1→4)
│   └── tjk_*.py          #   stage 1–8 + yardımcılar (tjk_betting, features_live, ...)
├── data/                 # TÜM .csv (ham veri + üretilen: predictions_log, oof, ...)
├── outputs/              # Günlük üretilen .md (predictions_*, bets_*, live_performance)
├── docs/                 # Raporlar (RAPOR*.md) + tez/makale (PDF, docx)
├── models/               # Eğitilmiş modeller (.pkl) + registry (.json)
├── reports/              # Akademik tablo/grafik + ablasyon & strateji özetleri
└── runs/                 # Streamlit iş kuyruğu log/durum dosyaları
```

> Tüm kod `src/` altında; her modül kendini `__file__`'den konumlandırıp veriyi
> `data/`, günlük çıktıları `outputs/` altında bulur. Bu yüzden scriptler herhangi
> bir dizinden çalıştırılabilir (`python src/tjk_pipeline.py`). Modül adları stage
> numaralı tutuldu (import'lar bunlara bağlı).

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
python src/tjk_pipeline.py

# Belirli aşamadan başla / tek aşama
python src/tjk_pipeline.py --from 3        # Stage 3 + 4
python src/tjk_pipeline.py --only 4        # Sadece modelleme

# Modelleme — tam model
python src/tjk_stage4_modeling.py

# Modelleme — piyasa sinyali ablasyonu (Ganyan_* olmadan)
python src/tjk_stage4_modeling.py --ablation

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

## Canlı Forward-Test ve Sürekli Öğrenme (Faz 2)

Yarışlar oynanmadan günlük tahmin üret, sonra gerçek sonuçla karşılaştır. Ayrıntılı gerekçe ve
mimari için bkz. **[docs/RAPOR_FAZ2_CANLI_TEST.md](docs/RAPOR_FAZ2_CANLI_TEST.md)**.

| Aşama | Dosya | İşlev | Çıktı |
|-------|-------|-------|-------|
| 5 | `tjk_stage5_live_program.py` | Günlük PROGRAM scraper (koşacak atlar) | `program_tablo.csv` |
| — | `tjk_features_live.py` | Canlı feature üretici (yalnız geçmiş veri) | (bellek içi) |
| 6 | `tjk_stage6_predict.py` | Tahmin: tam + ablation (Is_Winner/Is_Top3) | `predictions_log.csv`, `predictions_<date>.md` |
| 7 | `tjk_stage7_reconcile.py` | Tahmin ↔ gerçek sonuç → P@1/P@3/ROI | `live_performance.csv/.md` |
| — | `tjk_retrain_monitor.py` | Periyodik yeniden eğitim + drift izleme | `retrain_state.json`, `retrain_log.csv` |
| 8 | `tjk_stage8_betting_strategy.py` | Bahis stratejisi + kasa simülasyonu (Harville/Kelly) | `betting_strategy_backtest.csv`, `reports/bankroll_curve.png`, `bets_<date>.md` |
| — | `tjk_betting.py` | Bahis matematiği (Harville/EV/Kelly — saf kütüphane) | (içe aktarılır) |

```bash
# 1) Production modelleri (tam + ablation) hazır olmalı:
python src/tjk_stage4_modeling.py            # tam → production_registry.json
python src/tjk_stage4_modeling.py --ablation  # ablation → production_registry_ablation.json

# 2) Günlük döngü:
python src/tjk_stage5_live_program.py --headless   # bugünün programını çek
python src/tjk_stage6_predict.py                    # tahmin üret (tam + ablation)
#   ... yarışlar oynanır ...
python src/tjk_pipeline.py --only 1                 # o günün sonuçlarını çek
python src/tjk_stage7_reconcile.py                  # canlı P@1/P@3/ROI

# 3) Sürekli öğrenme:
python src/tjk_retrain_monitor.py --status          # yeniden eğitim gerekli mi?
python src/tjk_retrain_monitor.py --run             # gerekliyse veri güncelle + yeniden eğit
```

**Zamanlama (cron, macOS/Linux):**
```cron
# Her gün 09:00 — program + tahmin
0 9 * * *  cd /path/Ganyan && .venv/bin/python src/tjk_stage5_live_program.py --headless && .venv/bin/python src/tjk_stage6_predict.py
# Her gün 23:00 — sonuç çek + değerlendir
0 23 * * * cd /path/Ganyan && .venv/bin/python src/tjk_pipeline.py --only 1 && .venv/bin/python src/tjk_stage7_reconcile.py
# Her Pazartesi 03:00 — gerekirse yeniden eğit
0 3 * * 1  cd /path/Ganyan && .venv/bin/python src/tjk_retrain_monitor.py --run
```

> **Not:** Stage 5 program sayfası JS-render olduğundan seçiciler ilk canlı çalıştırmada
> gözlemlenip gerekirse `parse_program_table` içinde ince ayar yapılmalıdır. Stage 1'in tarih
> aralığı (`tjk_scraper_stage1.py` içinde) ileri tarihler için güncellenmelidir.

## Bahis Stratejisi & Kasa Optimizasyonu (Faz 3)

Modelin per-at olasılıklarını **bahis kararına** çevirir: hangi koşuda hangi bahis türü
(Ganyan, Plase, İkili, Sıralı İkili, Üçlü, Tabela + çoklu-koşu) ne kadar oynanmalı? Yöntem
literatürde kurulu: **Harville/Plackett-Luce** (kombinasyon olasılığı) + **Benter edge**
(piyasa-üstü pozitif-EV) + **Kelly** (kasa yönetimi). Ayrıntı: **[docs/RAPOR_FAZ3_BAHIS_STRATEJISI.md](docs/RAPOR_FAZ3_BAHIS_STRATEJISI.md)**.

```bash
# Sızıntısız geçmiş olasılıklar (TimeSeriesSplit OOF)
python src/tjk_stage4_modeling.py --dump-oof          # → oof_predictions.csv
# Backtest (flat vs fractional Kelly kasa eğrisi)
python src/tjk_stage8_betting_strategy.py --backtest
# Günlük öneri
python src/tjk_stage8_betting_strategy.py --date 2026-06-20
```

> ⚠️ Egzotik geçmiş ödemeleri veride yok → backtest ödemeyi **piyasa-ima** ile tahmin eder
> (göreli-edge simülasyonu, literal TL değil). Forward'da gerçek ödemeler `payouts_tablo.csv`'ye
> kazınır. Yüksek varyans; akademik/araştırma amaçlı, gerçek bahis önerilmez.

## Arayüz (Streamlit)

Günlük işlemleri tek ekrandan yapmak için yerel panel:

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

Tarayıcıda açılan panelde 5 sekme var:
- **Bugün / Tahminler** — yarış-yarış model favorileri (full + ganyansız) ve piyasa favorisi;
  Kazanan (Ganyan) ↔ Tabela (ilk 3) görünüm seçimi
- **Performans** — kümülatif P@1/P@3/ROI ve günlük trend
- **Strateji** — günlük bahis önerileri (bahis türü, kombinasyon, pay, EV) + backtest kasa eğrisi
- **Modeller** — production registry, model karşılaştırması, grafikler
- **İşlemler** — *Programı Çek → Tahmin Üret → Strateji Üret → Sonuç Çek + Değerlendir →
  Strateji Backtest → Yeniden Eğit* butonları (mevcut scriptleri arka planda çalıştırır, canlı log)

## Lisans

(Belirtilecek — örn. MIT veya akademik kullanım.)
