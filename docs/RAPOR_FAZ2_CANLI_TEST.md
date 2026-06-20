# Faz 2 — Canlı Forward-Test ve Sürekli Öğrenme (Continual Learning)

**Tarih:** 2026-06-20
**Önkoşul:** Faz 1'de model metodolojisi düzeltildi ve savunulabilir hâle getirildi (bkz.
[RAPOR.md](RAPOR.md)). Bu rapor, modeli **gerçek dünyada** doğrulamak ve zamanla geliştirmek için
tasarlanan mimariyi ve bunun ardındaki bilimsel gerekçeyi belgeler.

---

## 1. Amaç

İki hedef var:

1. **Forward-test (ileriye dönük / kâğıt-üzeri test):** Yarışlar **oynanmadan önce** günlük tahmin
   üret, yarış bittikten sonra gerçek sonuçla karşılaştır. Modelin gerçek koşullardaki başarısını
   ölç.
2. **Sürekli öğrenme (continual learning):** Yeni sonuçlar biriktikçe modeli periyodik **yeniden
   eğit** ve canlı performansı **izle**; böylece model zamanla veriye ve değişime (drift) uyum
   sağlasın.

> ⚠️ **Etik/kapsam notu:** Bu sistem akademik/araştırma amaçlıdır ve **kâğıt-üzeri** (sanal bahis)
> test için tasarlanmıştır. Gerçek parayla bahis kararına dayanak yapılması önerilmez; ROI
> sonuçları yüksek varyans içerir.

---

## 2. "Hatalarından Öğrenen AI Agent" — Gerçekçi Olan vs Hype Olan

Projenin başında "AI agent hatalarından ders alarak daha doğru tahmin etsin" fikri gündeme geldi.
Bu fikrin tez açısından **doğru çerçevelenmesi kritiktir**; çünkü iki çok farklı şeyi kastedebilir:

### 2.1. Gerçekçi ve doğru olan (BU PROJEDE SEÇİLEN)
- **Sürekli öğrenme = yeniden eğitim:** Her gün yeni yarış sonuçları geldikçe eğitim verisi büyür;
  model periyodik olarak yeniden eğitilir. "Öğrenme" budur — daha çok veri + sezona/yeni
  atlara/jokeylere uyum. Standart, kanıtlanmış MLOps yaklaşımı.
- **İzleme + drift tespiti:** Canlı performans (P@1, ROI) izlenir; düşüş olursa yeniden eğitim
  tetiklenir.
- **Bir "agent"ın doğru rolü:** Tahmini kendi başına "sezmek" değil, **operasyon döngüsünü
  otomatikleştirmek** — programı çek, tahmin üret, sonucu kaydet, metrik hesapla, ne zaman yeniden
  eğitileceğine karar ver.

### 2.2. Hype olan (BİLİNÇLİ OLARAK SEÇİLMEDİ)
"Bir LLM, yanıldığı yarışlara bakıp *neden yanıldığını düşünerek* kendini düzeltsin ve daha iyi
tahmin etsin" yaklaşımı çekici görünür ama bu problem için **yanıltıcıdır**:

1. **At yarışı indirgenemez varyans içerir.** Sonuçlar büyük ölçüde rastlantısaldır; mükemmel bir
   model bile yarışların çoğunu bilemez. "Öğrendikçe %90 isabet" gibi bir tavan **yoktur**.
   Bahis piyasası (ganyan) zaten verimliye yakındır; P@1 ~%38–40 pratik tavana yakın ve piyasayı
   geçmektedir.
2. **LLM, sayısal olasılıkları gradient-tabanlı eğitimden daha iyi kalibre edemez.** "Şu yarışı
   kaçırdım, ayarlayayım" türü akıl yürütme, kalibre olasılık üretmez; büyük olasılıkla **uydurma
   örüntülere (gürültüye overfit)** yol açar.
3. **Reaktif kurcalama modeli bozar.** Kaybeden serilere tepki vererek sürekli "düzeltme" yapmak,
   istatistiksel gürültüyü kovalamaktır. **Disiplinli, sabit kadanslı yeniden eğitim** > reaktif
   müdahale.

**Sonuç:** Bu projede "öğrenme", veri büyüdükçe **yeniden eğitim + izleme** olarak uygulanır.
LLM/agent katmanı (ileride eklenirse) yalnızca **orkestrasyon ve raporlama** yapar — tahmini
asla "sezmez".

---

## 3. Forward-Test Neden Altın Standart?

- **Sızıntı fiziksel olarak imkânsızdır:** Yarış oynanmadan tahmin yapıldığı için modelin geleceği
  görmesi mümkün değildir. Backtest'te her zaman kalan "acaba sızıntı mı var" şüphesi burada
  tamamen ortadan kalkar.
- **Prospektif kanıt:** Tez için en güçlü ifade — *"Modeli N hafta canlı koşullarda çalıştırdık;
  P@1=%X, ROI=%Y gözlemledik."* Bu, geriye dönük tablolardan çok daha ikna edicidir.
- **Gerçek dağılım kayması (drift) görünür:** Mevsim, pist değişimi, yeni atlar gibi etkiler ancak
  canlı testte ortaya çıkar.

---

## 4. Mimari (Veri Akışı)

```
                ┌─────────────────────────────────────────────────────────┐
                │  GEÇMİŞ VERİ (eğitim)                                     │
                │  yaris_ana_tablo · atlar_statik_tablo · idmanlar_tablo    │
                └───────────────┬─────────────────────────────────────────┘
                                │ Stage 3 (feature) + Stage 4 (model)
                                ▼
                   models/ (tam + ablation production)
                                │
   ┌────────────────────────────┼─────────────────────────────────────────┐
   │  GÜNLÜK CANLI DÖNGÜ         │                                          │
   │                            ▼                                          │
   │  Stage 5: PROGRAM scraper  →  program_tablo.csv (koşacak atlar)        │
   │                            │                                          │
   │  tjk_features_live: geçmiş veriyle 36 feature üret                     │
   │                            ▼                                          │
   │  Stage 6: TAHMİN (tam + ablation) → predictions_log.csv               │
   │                            │                                          │
   │            ⏳ yarışlar oynanır                                         │
   │                            ▼                                          │
   │  Stage 1 (o günün sonucu) → yaris_ana_tablo'ya eklenir                 │
   │                            ▼                                          │
   │  Stage 7: EŞLEŞTİR + METRİK → live_performance.csv (P@1/P@3/ROI)       │
   └────────────────────────────┬─────────────────────────────────────────┘
                                │  (N günde bir / drift olunca)
                                ▼
              tjk_retrain_monitor: Stage 1→2→3→4 yeniden eğit (tam+ablation)
                                │
                                └──────────► models/ tazelenir (döngü kapanır)
```

### Bileşenler
| Bileşen | Dosya | İşlev |
|---------|-------|-------|
| Program scraper | `tjk_stage5_live_program.py` | Koşacak atları çeker (Selenium, JS-render) |
| Canlı feature | `tjk_features_live.py` | Stage 3 mantığıyla 36 feature (yalnız geçmiş veri) |
| Tahmin | `tjk_stage6_predict.py` | Tam + ablation model → olasılık & sıralama |
| Eşleştirme | `tjk_stage7_reconcile.py` | Tahmin ↔ gerçek sonuç → P@1/P@3/ROI |
| Yeniden eğitim | `tjk_retrain_monitor.py` | Veri tazele + periyodik yeniden eğit + izle |

---

## 5. Neden Hem "Tam" Hem "Ganyansız" Tahmin?

İki farklı **tahmin penceresi** vardır:

- **Ganyansız (ablation) model → erken tahmin:** Kesin ganyan ancak koşu başında belli olur.
  Yarıştan saatler önce karar vermek isteyen biri için piyasasız model gerçekçidir. Faz 1
  ablasyonu, bu modelin AUC'unun tam modele çok yakın olduğunu (Δ~0.01–0.02) gösterdi.
- **Tam model → geç tahmin:** Sabah ganyanı oluştuktan sonra, koşuya yakın daha yüksek bilgili
  tahmin.

Canlı ortamda ikisi de loglanıp **hangisinin gerçek dünyada daha iyi olduğu** ölçülecek. Bu,
"piyasa sinyali ne kadar değerli?" sorusunun prospektif (ileriye dönük) cevabıdır — tez için güçlü.

---

## 6. Sürekli Öğrenme — Politika

- **Veri tazeleme:** Her gün (veya birkaç günde bir) yeni sonuçlar Stage 1 (resume/append) ile
  eklenir, yeni atlar için Stage 2, ardından Stage 3 master matrix yenilenir.
- **Yeniden eğitim kadansı:** Sabit periyot (öneri: haftalık) **veya** drift tetikli (rolling
  canlı P@1/ROI eşik altına düşerse). Sabit kadans, reaktif aşırı-uyumdan korur.
- **Doğrulama korunur:** Yeniden eğitimde aynı zaman-bazlı CV + per-fold imputation kullanılır;
  metrikler savunulabilir kalır.
- **Sürüm takibi:** Her yeniden eğitim yeni zaman damgalı model + registry kaydı üretir; geriye
  dönük karşılaştırma mümkün olur.

---

## 7. Riskler ve Sınırlar

- **Scraper kırılganlığı:** Program sayfası JS-render; seçiciler canlı DOM'a karşı iteratif
  geliştirilecek. TJK arayüz değiştirirse bakım gerekir.
- **Eksik özellikler:** Yeni/ilk-kez koşan atlarda handikap/idman/soy verisi olmayabilir; mevcut
  imputation + missingness flag'leri bunu karşılar (ablation modeli zaten ganyana bağımlı değil).
- **Düşük tavan:** Hiçbir model yarışların çoğunu bilemez; başarı ölçütü "piyasayı geçmek" ve
  "pozitif/temkinli ROI"dir, "çoğu yarışı bilmek" değil.
- **Varyans:** Kısa dönem ROI çok oynaktır; anlamlı sonuç için yeterli sayıda yarış-günü gerekir
  (öneri: en az birkaç hafta) ve güven aralığıyla sunulmalıdır.

---

## 8. Sonraki Adım (Uygulama)

**Faz 1 (öncelik):** Stage 5 → `tjk_features_live` → Stage 6 → Stage 7 + Stage 4'e ablation
production model kaydı. Bu, uçtan uca forward-test döngüsünü kurar.
**Faz 2:** `tjk_retrain_monitor.py` + zamanlama (cron/launchd) dokümantasyonu.

---

*Bu rapor, canlı test ve sürekli öğrenme mimarisinin tasarımını ve bilimsel gerekçesini, kod
yazımından önce belgelemektedir.*
