# RAPOR — FAZ 3: Bahis Stratejisi & Kasa Optimizasyonu (Stage 8)

> Bu rapor, modelin per-at olasılıklarını **bahis kararına** dönüştüren karar
> motorunu belgeler: hangi koşuda hangi bahis türü (Ganyan, Plase, İkili, Sıralı
> İkili, Üçlü, Tabela + çoklu-koşu) ne kadar oynanmalı? Sabit kasayla flat ve
> fractional Kelly stratejileri kronolojik simüle edilir.

---

## 1. Motivasyon ve Soru

Faz 1–2'de model her at için **sızıntısız** kazanma (`prob_winner`) ve ilk-3
(`prob_top3`) olasılığı üretiyordu. Doğal bir sonraki adım: bu olasılıkları
**kâr amaçlı bir bahis politikasına** çevirmek.

Kullanıcının sorusu: *"Modelin elinde belli bir para olsun; oynadıkça kazandığı/
kaybettiğiyle en mantıklı oynama tarzını bulsun — bazı koşuda sırasız üçlü, bazıda
sıralı ikili, bazıda ganyan+ikili."*

**Cevap: Mantıklı ve yapılabilir** — ama "hatadan öğrenen RL ajanı" olarak değil.
Doğru çerçeve, finans/yarış literatüründe kurulu üç parçadır:

1. **Harville / Plackett-Luce** — tekil kazanma olasılıklarından sıralı/sırasız
   kombinasyon (egzotik) olasılıkları.
2. **Benter / Bolton-Chapman edge** — modelin olasılığı piyasanın ima ettiğinden
   yeterince yüksekse (pozitif EV) bahis.
3. **Kelly kriteri** — kasanın logaritmik büyümesini maksimize eden pay büyüklüğü.

RL küçük örnekte (yıllık ~2 bin koşu, egzotik isabet seyrek) gürültüye aşırı uyar;
bu yüzden bilinçli olarak EV-temelli politika + kasa simülasyonu tercih edildi.

---

## 2. Yöntem

### 2.1 Olasılık → Kombinasyon (Harville/Plackett-Luce)
Koşu-içi normalize edilmiş kazanma vektörü `p`'den:

| Bahis | Formül |
|-------|--------|
| Sıralı İkili (exacta) | `p_i · p_j/(1−p_i)` |
| İkili (quinella) | exacta(i,j) + exacta(j,i) |
| Üçlü (sıralı top-3) | `p_i · p_j/(1−p_i) · p_k/(1−p_i−p_j)` |
| Tabela (sıralı top-4) | Plackett-Luce zinciri (4 terim) |
| Plase | atın ilk-3'te bitme olasılığı (Harville) veya modelin `prob_top3`'ü |
| Çifte / Pick-N | bacakların kazanma olasılıkları çarpımı |

Tüm bunlar `tjk_betting.py`'de; öz-testte sıralı-ikili/üçlü olasılıklarının
koşu-içi toplamı ≈ 1, quinella ≥ exacta gibi tutarlılıklar doğrulanır.

### 2.2 Favori-uzunatış kalibrasyonu (λ)
Harville favorilerin egzotik olasılığını sistematik olarak **abartır**
(Lo-Bacon-Shor, 1995). Düzeltme: `p_i^λ / Σ p_j^λ`, λ<1 dağılımı düzleştirir.
Varsayılan **λ=0.85** (CLI: `--lam`).

### 2.3 Piyasa-ima ödeme (kritik veri kısıtı)
`yaris_ana_tablo.csv` yalnız **Ganyan** (kazanma) oranı + **Siralama** (bitiş)
içerir; egzotiklerin **geçmiş ödemeleri YOK**. Bu yüzden:

- **Backtest** ödemeyi piyasadan tahmin eder:
  `ödeme ≈ (1 − takeout) / P_market(kombinasyon)`,
  burada `P_market` aynı Harville'in **piyasa** kazanma olasılıklarına (Ganyan
  oranlarından) uygulanmasıyla bulunur.
- **Ganyan bahsi istisna**: gerçek Ganyan oranı kullanılır (veride mevcut).
- **Forward (canlı)**: yarış sonrası gerçek ödemeler `payouts_tablo.csv`'ye kazınır
  (Stage 1 `parse_payouts`) ve varsa onlar kullanılır.

> **Sonuç:** Backtest bir **göreli-edge simülasyonu**dur — "model piyasayı doğru
> yönde yenebiliyor mu?" sorusunu ölçer; literal TL P&L değildir. Bu, dürüstlük
> açısından açıkça raporlanır.

Takeout (komisyon) varsayımları (TJK yaklaşık): Ganyan %19, Plase %20,
İkili/Sıralı İkili %25, Üçlü %27, Tabela %30, çoklu-koşu %30.

### 2.4 EV ve bahis seçimi
Her koşuda, her bahis türü için modele göre ilk-N (varsayılan 4) at arasından
**en yüksek EV** kombinasyonu bulunur:
`EV/stake = P_model · ödeme − 1`. Eşik (varsayılan **+%5**) üstündekiler oynanır.
Takeout %25'te bu, modelin o kombinasyonda piyasadan **~%33+ yüksek** olasılık
biçmesini gerektirir → bilinçli yüksek bar, aşırı-bahsi önler.

### 2.5 Kasa yönetimi (flat vs fractional Kelly)
- **Flat:** her bahis = başlangıç kasasının sabit %1'i.
- **Fractional Kelly:** `f* = p − (1−p)/b` (b = net oran), **¼-Kelly** ölçekli,
  tek bahiste kasanın en çok **%5'i** (cap). Varyansı düşürür.
İkisi de **aynı** seçilmiş bahis kümesinde, kronolojik sırada paralel simüle
edilir → adil kıyas. Çıktı: `reports/bankroll_curve.png`.

### 2.6 Çoklu-koşu (deneysel)
Çifte (2 bacak) ve 3'lü Ganyan (3 bacak) banker (her bacakta top-1) biletleri
ile EV hesaplanır. **Çok yüksek varyans** → ayrı/etiketli raporlanır, ana kasa
simülasyonuna karıştırılmaz.

---

## 3. Mimari ve Dosyalar

| Bileşen | Dosya | İşlev |
|---------|-------|-------|
| Matematik | `tjk_betting.py` | Harville, kalibrasyon, EV, Kelly (saf, test edilebilir) |
| OOF ihracı | `tjk_stage4_modeling.py --dump-oof` | Sızıntısız geçmiş olasılıklar → `oof_predictions.csv` |
| Strateji motoru | `tjk_stage8_betting_strategy.py` | Backtest + günlük öneri |
| Gerçek ödeme | `tjk_scraper_stage1.py` (`parse_payouts`) | `payouts_tablo.csv` (forward) |
| Arayüz | `app.py` → **💰 Strateji** sekmesi | Öneri + kasa eğrisi |

### Veri akışı
```
master_feature_matrix.csv
        │  (TimeSeriesSplit OOF — sızıntısız)
        ▼
oof_predictions.csv ──► tjk_stage8 --backtest ──► betting_strategy_backtest.csv
                                                   reports/betting_strategy_summary.md
                                                   reports/bankroll_curve.png

predictions_log.csv  ──► tjk_stage8 --date GÜN  ──► bets_<date>.md  (günlük öneri)
```

---

## 4. Kullanım

```bash
# 1) Sızıntısız geçmiş olasılıklar (production model, TimeSeriesSplit OOF)
python tjk_stage4_modeling.py --dump-oof        # → oof_predictions.csv

# 2) Backtest (flat vs Kelly kasa eğrisi + bahis-türü kırılımı)
python tjk_stage8_betting_strategy.py --backtest

# 3) Günlük öneri (canlı tahminlerden)
python tjk_stage8_betting_strategy.py --date 2026-06-20

# Parametreler: --lam 0.85  --ev 0.05  --bankroll 1000
```

UI: `streamlit run app.py` → **💰 Strateji** sekmesi.

---

## 5. Dürüstlük Sınırları (özet)

1. **Egzotik geçmiş ödeme yok** → backtest piyasa-ima tahminiyle çalışır
   (göreli-edge, literal TL değil). Forward'da gerçek ödeme kazınır.
2. **Varyans acımasız** — egzotikler nadiren tutar, büyük öder. Anlamlı egzotik
   ROI için **binlerce** bahis gerekir; 18 koşuluk forward-test ROI = gürültü.
3. **Harville yanlılığı** λ ile törpülenir ama "pozitif EV" iyimser olabilir.
4. **Takeout %20–30** büyük negatif sürükleme; sürekli kâr gerçek bir edge ister.
5. **Parimutuel refleksivite:** gerçek bahiste pay havuzu oranları kaydırır
   (küçük TJK egzotik havuzlarında belirgin); kâğıt-üzeri simülasyonda yok sayılır.
6. **Amaç akademik/araştırma.** Gerçek bahis önerilmez.

---

## 6. İlk Sonuçlar (1.830 koşu, sızıntısız OOF)

### 6.1 Dürüst headline — Model vs Piyasa seçim isabeti (ödemesiz)
Bu tablo **ödeme varsaymadan, dairesel olmadan** ölçer: modelin doğal seçimi
(olasılığa göre ilk-k) piyasanın doğal seçiminden (favori sırası) daha sık mı
tutuyor? Tüm bahis türlerinde model **pozitif** fark veriyor:

Fark şans mı? **McNemar exact testi** (aynı 1.830 koşuda eşleşmiş ikili sonuç)
ile sınanır:

| Bahis | n | Model isabet | Piyasa isabet | Δ (pp) | McNemar p | Anlamlı? |
|-------|---|--------------|---------------|--------|-----------|----------|
| Ganyan | 1.830 | **39.8%** | 31.7% | **+8.1** | 1.6e-16 | ✓✓✓ |
| İkili (sırasız 2) | 1.830 | 22.1% | 16.4% | +5.7 | 1.1e-10 | ✓✓✓ |
| Sıralı İkili | 1.830 | 14.6% | 9.6% | +5.0 | 2.5e-10 | ✓✓✓ |
| Plase (ilk-3) | 1.830 | 73.0% | 68.3% | +4.7 | 1.9e-06 | ✓✓✓ |
| Üçlü (sıralı 3) | 1.830 | 4.0% | 2.7% | +1.3 | 1.4e-02 | ✓ |
| Tabela (sıralı 4) | 1.830 | 1.6% | 0.9% | +0.7 | 4.1e-02 | ✓ |

→ Model **her türde** piyasa favorisinden daha isabetli seçiyor ve fark **tüm
türlerde istatistiksel olarak anlamlı** (p<0.05). Güçlü türlerde (Ganyan, İkili,
Sıralı İkili, Plase) p≪0.001 → fark kesinlikle şans değil; nadir türlerde (Üçlü,
Tabela) sınırda anlamlı (örneklem azaldıkça güç düşer). Bu, "hangi bahis türünde
değer var?" sorusunun **çıkarımsal** (sadece betimsel değil) cevabıdır.

### 6.2 Ganyan kasası — gerçek oran (tek güvenilir bankroll)
178 pozitif-EV Ganyan bahsi (gerçek oranla), isabet %39.9, **flat ROI +132%**.

> ⚠️ **KRİTİK DÜRÜSTLÜK:** Bu backtest **iyimser**. Aynı modelin **canlı
> forward-testinde** Ganyan ROI **~ −36%** çıktı (18 koşu). Geçmiş OOF backtest
> geleceği garanti ETMEZ — gerçek hakem forward-test'tir. Backtest/forward uçurumu
> (neden +132% vs −36%?) tam da forward-test disiplininin neden zorunlu olduğunu
> gösterir: tarihsel veriye dayalı ROI sistematik olarak şişer.

### 6.3 Egzotik EV tablosu — neden literal TL DEĞİL
Egzotik geçmiş ödemesi olmadığından ödeme `(1−takeout)/P_market` ile tahmin edilir;
skorlamada da aynı tahmin kullanılır → **dairesel**. Modelin edge'i, seçtiği
kombinasyonların `P_market`'in ima ettiğinden daha sık tutmasına yol açar; bu kapalı
döngü "imkânsız kâr" üretir (ilk denemede Kelly kasası ~1e42'ye patladı). Bu yüzden
egzotik ROI **fantazidir** ve bankroll olarak SUNULMAZ; yalnız (6.1) tanısı ve
forward-test gerçek kanıttır. Korumalar (ödeme tavanı 50x, kombinasyon olasılığı
tabanı %2, koşu başına ≤2 bahis) eklenmiştir ama dairesellik yapısaldır.

### 6.5 Olasılık kalibrasyonu (backtest/forward uçurumunu açıklar)
`python reports/generate_calibration.py` ile OOF olasılıkları reliability diagram +
Brier + ECE ile ölçüldü:

| Hedef | Brier | ECE | Taban oran | Durum |
|-------|-------|-----|-----------|-------|
| **Is_Winner** | 0.076 | **0.005** | 10.2% | ~mükemmel kalibre |
| **Is_Top3** | 0.263 | **0.302** | 30.7% | ciddi **over-confident** |

**Bulgu:** Kazanma olasılıkları neredeyse mükemmel kalibre (ECE 0.005), ama ilk-3
olasılıkları **aşırı-güvenli** (ECE 0.30). Sebep: production modeller sınıf-dengeleme
(`is_unbalance` / `auto_class_weights` / `scale_pos_weight`) kullanır — bu, sıralamayı
(AUC, P@1) güçlendirir ama olasılıkları gerçek taban orandan uzaklaştırır. İlk-3
modelinde bu distorsiyon büyük.

**Sonuç (uçurum yorumu):** Egzotik EV hesabı `prob_top3` türevlerine dayandığından,
over-confident olasılıklar EV'yi şişirir → backtest'in iyimserliğinin (6.2/6.3) bir
kaynağı budur. Ranking güçlü; ama **betting EV için olasılıklar isotonic/Platt ile
yeniden kalibre edilmeli** (sıralama metrikleri değişmez, EV gerçekçileşir). Bu, doğal
bir sonraki iyileştirme adımıdır (bu fazda yalnız ölçüldü, uygulanmadı).
Grafikler: `reports/calibration_is_winner.png`, `reports/calibration_is_top3.png`.

### 6.6 Sonuç
- **Savunulabilir bulgu:** Model, piyasa favorisinden tüm bahis türlerinde daha iyi
  *seçim* yapıyor (6.1) ve fark **istatistiksel olarak anlamlı** (McNemar p<0.05;
  güçlü türlerde p≪0.001) — özellikle Ganyan +8.1pp (p=1.6e-16), İkili +5.7pp.
- **Kalibrasyon:** Kazanma olasılıkları kalibre, ilk-3 over-confident (6.5) → egzotik
  EV iyimserliğinin teknik kaynağı; recalibration bir sonraki adım.
- **Bankroll iddiası YOK:** Tarihsel ROI (Ganyan +132%, egzotik dairesel) gelecek
  kârını göstermez. Gerçek kâr ölçümü yalnız gerçek-ödemeli forward-test ile yapılır
  (`payouts_tablo.csv` ileriye dönük toplanıyor).
