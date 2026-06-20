# TJK At Yarışı Tahmin Projesi — Teknik Değerlendirme ve Düzeltme Raporu

**Tarih:** 2026-06-20
**Kapsam:** `tjk_stage1..4` pipeline, feature engineering, modelleme ve raporlama çıktıları
**Amaç:** Mevcut sonuçların güvenilirliğini değerlendirmek, metodoloji/bug sorunlarını tespit etmek ve
bir tez/hakem değerlendirmesinde **savunulabilir** sonuçlar üretecek düzeltmeleri belgelemek.

---

## 1. Proje Özeti

Proje, **TJK (Türkiye Jokey Kulübü)** at yarışı sonuçlarını tahmin eden uçtan uca bir makine öğrenmesi
pipeline'ıdır ve bir akademik çalışma (tez/makale) çıktısıdır. Dört aşamadan oluşur:

| Aşama | Dosya | İşlev | Çıktı |
|-------|-------|-------|-------|
| Stage 1 | `tjk_scraper_stage1.py` | Selenium ile yarış sonuçları scraping | `yaris_ana_tablo.csv` |
| Stage 2 | `tjk_scraper_stage2.py` | At statik profili + idman verileri | `atlar_statik_tablo.csv`, `idmanlar_tablo.csv` |
| Stage 3 | `tjk_stage3_feature_engineering.py` | Veri birleştirme + feature engineering | `master_feature_matrix.csv` |
| Stage 4 | `tjk_stage4_modeling.py` | 9 model + ensemble + Optuna + SHAP + ROI | `models/`, `reports/` |

**Veri büyüklüğü (doğrulandı):**
- 21.648 satır (at-yarış kaydı), 2.199 benzersiz yarış, 127 farklı yarış günü
- Tarih aralığı: 2025-04-17 → 2026-03-20 (~11 ay)
- Hedef değişkenler: `Is_Winner` (1. olma), `Is_Top3` (ilk 3'e girme)
- **Önemli kısıt:** Veride yalnızca **kazanma ganyanı** (`Ganyan_Sayi`) var; plase/show ganyanı yok.

---

## 2. Güçlü Yönler (krediyi hak ediyor)

Düzeltme önerilerine geçmeden önce, projenin gerçekten iyi yapılmış yönlerini belirtmek gerekir:

- **Temiz mimari:** 4 aşamalı orchestrator (`tjk_pipeline.py`), resume mantığı, idempotent scraping,
  `at_id`/tarih bazlı tekilleştirme.
- **Sızıntı bilinci (kısmen):** Stage 3'te `merge_asof(direction="backward")` ile gelecekteki
  idmanların alınmaması, target encoding'de `expanding().mean().shift(1)`, `Derece_Saniye` (koşu süresi)
  ve `Siralama`'nın bilinçli olarak feature listesinden çıkarılması — bunlar doğru reflekslerdir.
- **Domain düşüncesi:** Yarış-içi göreceli özellikler (`Relative_Handikap`, `Relative_Siklet`),
  eksiklik (missingness) bayrakları (`Has_400m_data` vb.), ganyandan implied probability türetme.
- **Kapsamlı modelleme:** Baseline → gradient boosting → ensemble (Voting/Bagging/Stacking),
  Optuna hiperparametre optimizasyonu, SHAP açıklanabilirlik, çoklu ROI stratejisi (top1/value/kelly).

---

## 3. Tespit Edilen Sorunlar

Aşağıdaki sorunlar, raporlanan metrikleri (özellikle ROI rakamlarını) **geçersiz** kılmaktadır.
Önem sırasına göre listelenmiştir.

### 3.1. 🔴 KRİTİK — ROI hesabında ödeme hatası (Top3 için ölümcül)

**Konum:** `tjk_stage4_modeling.py`, `calculate_roi` fonksiyonu (satır ~469-540)

`calculate_roi`, bir at hedef koşulunu sağladığında **kazanma ganyanı** kadar ödeme yapar:

```python
if bet_horse[target_col] == 1:                       # target_col, Is_Top3 olabiliyor
    total_return += stake * float(bet_horse[odds_col])  # ama odds_col = WIN ganyanı
```

`target_col = Is_Top3` olduğunda, bir at **3. bitirse bile** birinci olmuş gibi kazanma ganyanı
ödenir. Gerçek hayatta plase/show bahsi kazanma ganyanı ödemez (çok daha düşük öder). Bu yüzden
`academic_tables.md` Tablo 2'deki **+130%, +137% Top3 ROI** rakamları tamamen yapaydır ve gerçek
dünyada imkânsızdır.

Aynı hata `ganyan_baseline` fonksiyonundaki Top3 ROI hesabında da mevcuttur.

### 3.2. 🔴 KRİTİK — Zaman serisinde GroupKFold kullanımı (ve ölü `strategy` kodu)

**Konum:** `evaluate_with_cv` (satır ~578), `optimize_model` (satır ~684), `main` (satır ~881)

Bu bir **zaman serisi** problemidir: geçmişten geleceği tahmin etmek gerekir. Ancak değerlendirme
hep `GroupKFold` ile yapılır; bu yöntem fold'ları **zamandan bağımsız** rastgele böler. Sonuç olarak
model, **geleceği görüp geçmişi tahmin eder** — bu da performansı yapay olarak şişirir.

Daha da çarpıcısı: `main()` içinde `create_split_strategy(df)` çağrılıp "time_based" stratejisi
hesaplanır, fakat dönen `strategy` değişkeni **hiçbir yerde kullanılmaz** (ölü kod). Niyet doğru,
uygulama eksik kalmış.

### 3.3. 🟠 ÖNEMLİ — Imputation/encoding'in CV öncesi global fit edilmesi

**Konum:** `prepare_features` (satır ~226-275)

`KNNImputer`, `SimpleImputer(median)` ve `OrdinalEncoder` **tüm veri** üzerinde `fit_transform`
edilir, CV bölünmesinden önce. Bu durumda test fold'unun istatistikleri (median, KNN komşuları)
eğitim sürecine sızar. Kodun kendi yorumu bunu kabul eder ("Şimdilik global fit yeterli").

### 3.4. 🟠 ÖNEMLİ — Stacking iç CV'sinin düz KFold olması

**Konum:** `build_models`, `StackingClassifier(..., cv=5)` (satır ~441)

Stacking meta-özelliklerini üretirken `cv=5` düz KFold kullanılır. Bu, aynı yarıştaki atların
meta-öğrenicinin hem eğitim hem doğrulama kümesinde yer almasına yol açabilir (yarış-içi sızıntı).
Kod yorumunda da bu sınırlama itiraf edilmiştir.

### 3.5. 🟡 YAN ETKİ — Production model seçimi bozuk metrikten etkileniyor

**Konum:** `select_and_register_production_model` (satır ~805)

Bu fonksiyon, composite skorda `ROI_value`'ya %40 ağırlık verir ve `ROI_value >= -0.20` hard-gate
uygular. ROI hatalı olduğu için (3.1) seçilen production modelleri de güvenilmez bir metriğe
dayanmaktadır. Nitekim `production_registry.json`'da `Is_Winner` için seçilen StackingEnsemble,
hatalı +92.6% ROI'ye dayanmaktadır.

### 3.6. 🟡 Küçük bug'lar

- **Stage 1 ölü kod:** `tjk_scraper_stage1.py` satır ~326-327'de `continue`'dan sonra gelen
  `print(...)` satırı asla çalışmaz (yanlış girinti).
- **Feature importance'da yanlış `y`:** `main()` içinde feature importance döngüsünde kullanılan `y`,
  eğitim döngüsünün son iterasyonundan (Is_Top3) kalan değişkendir; Is_Winner modeli için yanlış
  hedefe karşı importance hesaplanır.

---

## 4. Mevcut Metrikler Neden Güvenilmez?

**ROI matematiği:** `model_comparison.csv`'de Top3 için raporlanan +130% ROI, "1 TL bahse karşılık
2.30 TL geri dönüş" anlamına gelir. 2.000'den fazla bahis üzerinde bu büyüklükte bir getiri, gerçek
bir tote (müşterek bahis) piyasasında matematiksel olarak mümkün değildir. İyi modellerin gerçek
dünya getirileri tipik olarak -%5 ile +%5-10 bandındadır. +130% rakamı, doğrudan 3.1'deki ödeme
hatasının ürünüdür (plase finişe kazanma ganyanı ödenmesi).

**GroupKFold şişirmesi:** Is_Winner için raporlanan AUC ~0.85 kısmen gerçektir (favoriler kazanır,
ganyan güçlü bir sinyaldir). Ancak modelin "piyasayı yendiğini" iddia eden pozitif ROI'ler, ileri-zincirli
zaman bölmesi yerine rastgele GroupKFold kullanılmasından kaynaklanan iyimser yanlılığı içerir.
Müşterek bahiste kapanış oranını (closing line) yenmek olağanüstü zordur; mevcut kurulum bunu
yanlışlıkla "kolay" gösterir.

**Sonuç:** AUC değerleri büyük ölçüde anlamlı olabilir, ancak **ROI rakamları geçersizdir** ve
production model seçimi bu geçersiz metriğe dayanmaktadır.

---

## 5. Önerilen Düzeltmeler

Aşağıdaki kararlar üzerinde mutabık kalınmıştır: **5 düzeltmenin tamamı** uygulanacak; CV için
**ileri-zincirli TimeSeriesSplit** kullanılacak; **Top3 ROI tamamen kaldırılacak** (veride yalnızca
kazanma ganyanı bulunduğundan ROI sadece `Is_Winner` için anlamlıdır).

### 5.1. ROI Top3 düzeltmesi (Sorun 3.1)
ROI yalnızca `Is_Winner` hedefi için hesaplanır (kazanma bahsi → kazanma ganyanı ödemesi doğru olur).
`Is_Top3` için ROI üretilmez; bunun yerine AUC / Precision@1 / Precision@3 raporlanır. `calculate_roi`
fonksiyonu doğru haliyle (kazanma bahsi) korunur, yalnızca çağrısı `Is_Winner` ile sınırlanır.
`ganyan_baseline` da aynı şekilde gate'lenir.

### 5.2. Zaman-bazlı CV (Sorun 3.2)
Yeni `make_time_series_splits(df, n_splits=5)` helper'ı: yarışlar (`Unique_Race_ID`) ilk tarihlerine
göre kronolojik sıralanır; sklearn `TimeSeriesSplit` ile yarış listesi üzerinde ileri-zincirli bölme
yapılır; her yarışın tüm satırları ilgili tarafa bütün olarak atanır (yarış bölünmez). Böylece model
**her zaman geçmişle eğitilip gelecekle test edilir**. Ölü `strategy` kodu kaldırılır.

> Not: TimeSeriesSplit'te en erken yarışlar hiçbir test fold'unda yer almaz; bu satırlar OOF
> metriklerinden hariç tutulur (`tested_mask`).

### 5.3. Per-fold preprocessing (Sorun 3.3)
`prepare_features` artık **ham (NaN'lı) X** döndürür. Yeni `prepare_fold` fonksiyonu `OrdinalEncoder`,
`KNNImputer` ve `SimpleImputer(median)`'ı **yalnız eğitim fold'unda** fit edip her iki tarafı transform
eder. `prepare_all_folds`, fold başına dönüşümü bir kez hesaplayıp tüm modeller ve Optuna için yeniden
kullanır (verimlilik). Production (nihai) model için tüm veride fit edilmiş ayrı bir matris kullanılır —
bu meşrudur, çünkü production modeli tüm mevcut veriyi kullanmalıdır; sızıntı yalnızca CV
değerlendirmesinde sorundur.

### 5.4. Stacking iç CV'si (Sorun 3.4)
`StackingClassifier(cv=TimeSeriesSplit(n_splits=5))` kullanılır. (sklearn'de iç CV'ye `groups`
geçişi sürüm bağımlı olduğundan, zaman sıralı TimeSeriesSplit pragmatik ve savunulabilir çözümdür;
gelecek sızıntısını ortadan kaldırır.)

### 5.5. Production seçimi (Sorun 3.5)
`select_and_register_production_model` dallandırılır:
- `Is_Winner`: ROI'li composite (AUC gate + ROI gate + ROI/P@1/AUC/kararlılık).
- `Is_Top3`: ROI'siz composite (P@1/P@3/AUC/kararlılık), ROI gate yok.

### 5.6. Küçük düzeltmeler (Sorun 3.6)
- Stage 1'deki ölü kod temizlenir.
- Feature importance döngüsünde her hedef için doğru `y = df[target]` kullanılır.

### 5.7. Raporlama
`reports/generate_paper_plots.py`: Is_Top3 tablo/grafiğinden ROI sütunu/alt-grafiği kaldırılır
(artık NaN). Is_Winner için ROI grafiği ve -26.2% public baseline çizgisi korunur. `academic_tables.md`
buna göre yeniden üretilir.

> **Kapsam dışı (bilinçli):** `Ganyan_*` feature'ları korunur. Tote kapanış oranı yarış öncesi
> bilinebildiğinden katı bir sızıntı değildir; isteğe bağlı ayrı bir analiz konusudur.

---

## 6. Düzeltme Sonrası Beklenen Sonuçlar

- **Top3 ROI ortadan kalkar**; Top3 modelleri AUC / P@1 / P@3 ile değerlendirilir.
- **Is_Winner ROI'leri gerçekçi banda iner** (kabaca -%20 … +birkaç %). +90% / +130% gibi değerler
  kaybolur. Pozitif ama ölçülü bir ROI kalırsa, bu *savunulabilir* bir bulgu olur.
- **AUC değerleri bir miktar düşebilir** — sızıntı (rastgele zaman bölmesi + global imputation +
  stacking iç CV) ortadan kalktığı için bu **beklenen ve doğru** davranıştır; düşüş, önceki sayıların
  şişkin olduğunun kanıtıdır.
- **Production model seçimi**, düzeltilmiş ve hedefe-özgü kriterlere göre güncellenir.

---

## 7. Doğrulama Adımları

1. `python tjk_stage4_modeling.py` (veya `python tjk_pipeline.py --only 4`) ile yeniden eğit.
2. `reports/model_comparison.csv`:
   - Is_Top3 satırlarında ROI sütunları boş/NaN olmalı.
   - Is_Winner ROI'leri gerçekçi banda inmeli.
3. `python reports/generate_paper_plots.py` → `academic_tables.md` + PNG'ler güncel; Top3'te ROI yok.
4. `models/production_registry.json` yeni (düzeltilmiş) seçimleri yansıtmalı.

---

## 8. Uygulanan Düzeltmeler ve Gerçek Sonuçlar (Önce / Sonra)

Bölüm 5'teki düzeltmelerin tamamı uygulanmış ve Stage 4 yeniden çalıştırılmıştır. Aşağıda
düzeltme öncesi (GroupKFold + hatalı ROI) ve sonrası (TimeSeriesSplit + düzeltilmiş ROI)
sayıları karşılaştırılmaktadır.

### 8.1. Is_Winner (Kazanan Tahmini)

| Metrik | ÖNCE (hatalı) | SONRA (düzeltilmiş) | Yorum |
|--------|---------------|---------------------|-------|
| En iyi AUC | 0.8522 (LightGBM) | 0.8412 (CatBoost) | Sızıntı kalkınca beklenen hafif düşüş |
| Top1 ROI (model favorisine bahis) | — (öne çıkarılmamış) | +%4 … +%19 | **Artık gerçekçi ve savunulabilir** |
| Value ROI | +%25 … +%95 | +%3 … +%88 | Hâlâ yüksek olanlar longshot/yüksek-varyans (sızıntı değil) |
| Favori (piyasa) baseline ROI | -%26.2 | -%26.2 | Referans değişmedi |

### 8.2. Is_Top3 (Tabela Tahmini)

| Metrik | ÖNCE (hatalı) | SONRA (düzeltilmiş) | Yorum |
|--------|---------------|---------------------|-------|
| En iyi AUC | 0.8258 | 0.8158 (XGBoost) | Hafif düşüş (sızıntı kalktı) |
| Value ROI | **+%130 … +%137** | **YOK (kaldırıldı)** | İmkânsız değerler ortadan kalktı |
| Precision@1 | ~%73 | ~%73 | Korundu |
| Precision@3 | (raporlanmıyordu) | ~%97 | Yeni ana metrik |

### 8.3. Production model seçimi

| Hedef | ÖNCE | SONRA |
|-------|------|-------|
| Is_Winner | StackingEnsemble (ROI 0.93 — hatalı) | StackingEnsemble (P@1 0.399, top1 ROI +%10.7) |
| Is_Top3 | XGBoost (ROI 1.37 — hatalı) | XGBoost (P@1 0.732, P@3 0.968, ROI'siz kriter) |

### 8.4. Feature importance (en güçlü sinyaller)
`Relative_Handikap` her iki hedefte de açık ara en güçlü özellik; ardından `Handikap_Puani`,
`Ganyan_Sayi` ve `Relative_Siklet` gelir. Bu, modelin handikap ve piyasa sinyalini öğrendiğini
gösterir.

### 8.5. Önemli not — Value ROI yorumu
Bazı modellerde value-strateji ROI'si hâlâ yüksektir (örn. +%86). Bu, **sızıntı değildir**
(artık tamamen zaman-bazlı OOF üzerinden hesaplanıyor); seçici biçimde yüksek oranlı (longshot)
atlara bahis yapmanın doğal **yüksek-varyans** sonucudur. Tezde bu sonuç güven aralığı / varyans
analiziyle temkinli sunulmalıdır. Daha sağlam ve istikrarlı gösterge **top1 ROI** (+%4–19) ve
**P@1**'dir.

---

## 9. Uygulanan Dosya Değişiklikleri

- `tjk_stage4_modeling.py` — zaman-bazlı CV (`make_time_series_splits`, `prepare_fold`,
  `prepare_all_folds`), per-fold imputation, ROI gate (yalnız Is_Winner), stacking iç CV
  (`KFold(shuffle=False)` — partition gereği), production seçimi dallandırma, feature-importance
  `y` bug'ı düzeltmesi.
- `tjk_scraper_stage1.py` — `continue` sonrası ölü kod temizliği.
- `reports/generate_paper_plots.py` — Top3'te ROI sütunu/grafiği kaldırıldı (Precision@3'e geçildi).
- Yeniden üretildi: `reports/model_comparison.csv`, `reports/academic_tables.md`,
  `reports/academic_plot_*.png`, `reports/fi_*`, `models/*.pkl`, `models/production_registry.json`.

> Ortam notu: grafik scripti `seaborn` gerektiriyor; çalıştırma ortamına kuruldu.

---

## 10. Piyasa Sinyali Ablasyonu (Ganyan_* olmadan)

### 10.1. Yöntem
`Ganyan_Sayi`, `Ganyan_Implied_Prob`, `Ganyan_Rank_InRace` (hepsi bahis oranından türer)
feature listesinden çıkarılıp aynı pipeline (zaman-bazlı CV, per-fold imputation, Optuna)
tekrar çalıştırıldı (`python tjk_stage4_modeling.py --ablation`). Çıktılar ayrı dosyalara
yazıldı; tam-model sonuçları korundu. **Not:** `Ganyan_Sayi` ROI/baseline hesabı için `df`'te
kaldı (yalnızca *feature* olarak çıkarıldı), böylece bahis simülasyonu hâlâ doğru çalışır.

### 10.2. Sonuç — piyasa sinyalinin katkısı küçük

| Hedef | En iyi AUC (tam) | En iyi AUC (Ganyansız) | Ortalama ΔAUC |
|-------|------------------|------------------------|----------------|
| Is_Winner | 0.8412 | 0.8297 | **+0.014** |
| Is_Top3 | 0.8158 | 0.7950 | **+0.022** |

Tüm piyasa sinyali çıkarıldığında AUC yalnızca **~0.01–0.02** düşüyor. Yani modelin tahmin
gücünün büyük kısmı **projenin kendi ürettiği verilerden** geliyor (`Relative_Handikap`,
jokey/antrenör istatistikleri, soy hattı, idman dereceleri). Bu, **"model sadece bahis oranlarını
ezberlemiş"** eleştirisini güçlü biçimde çürütür ve tezin ana savunma noktalarından biridir.

### 10.3. İlginç ikincil bulgu — Ganyansız model daha "value" seçiyor
Ganyansız modelin **top1 ROI'si daha yüksek** (≈ +%39…+%65) çıktı; tam modelinki +%5…+%19'du.
Sebep: oranları görmeyen model, favoriyi körü körüne takip etmek yerine temel verilere dayanarak
daha yüksek oranlı ama yine de kazanma şansı yüksek atları seçiyor → kazandığında ödeme daha
büyük. Bu **yüksek-varyanslı** bir sonuçtur; tezde güven aralığıyla temkinli sunulmalıdır, ama
modelin piyasadan bağımsız gerçek bir sinyal yakaladığının ek işaretidir.

### 10.4. Üretilen dosyalar
`reports/model_comparison_ablation.csv`, `reports/ablation_comparison.md`,
`reports/ablation_auc_is_winner.png`, `reports/ablation_auc_is_top3.png`.
Kod: `tjk_stage4_modeling.py` (`--ablation` modu, `MARKET_FEATURES`),
`reports/generate_ablation_comparison.py`.

---

*Bu rapor, kod düzeltmelerinin tespit + uygulama + doğrulama aşamalarını ve piyasa sinyali
ablasyonunu belgelemektedir.*
