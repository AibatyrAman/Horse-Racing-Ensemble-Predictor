# Bahis Stratejisi — Backtest Özeti

**Kapsam:** 6,085 koşu (OOF, sızıntısız) • λ=0.85 • EV eşiği=+5%


## 1) Model vs Piyasa — seçim isabeti (ödeme VARSAYMADAN)

> En sağlam, dairesel-olmayan kanıt: modelin doğal seçimi (olasılığa göre ilk-k) piyasanın doğal seçiminden (favori sırası) daha sık mı tutuyor? Pozitif Δ → model o türde değer katıyor. **McNemar exact** testi (eşleşmiş ikili sonuç) farkın şans olup olmadığını ölçer: p<0.05 anlamlı.

| Bahis | n | Model isabet | Piyasa isabet | Δ (pp) | McNemar p |
|-------|---|--------------|---------------|--------|-----------|
| Ganyan | 6085 | 39.0% | 32.2% | +6.7 | 9.6e-32 ✓ |
| İkili | 6085 | 20.8% | 17.0% | +3.8 | 1.5e-14 ✓ |
| Sıralı İkili | 6085 | 12.0% | 9.3% | +2.7 | 6.0e-10 ✓ |
| Üçlü | 6084 | 3.6% | 2.9% | +0.6 | 1.6e-02 ✓ |
| Tabela | 6078 | 1.3% | 1.1% | +0.2 | 3.3e-01 |
| Plase | 6084 | 74.9% | 68.6% | +6.3 | 7.3e-30 ✓ |

## 2) Ganyan kasası — GERÇEK oran (tek güvenilir bankroll)

> Yalnız Ganyan'da gerçek ödeme (oran) var → bu kasa dairesel değil. Flat = sabit pay, Kelly = ¼-Kelly (cap %5).

- Bahis: **707** • isabet: **36.1%**
- Flat kasa: **1000 → 10016 TL** (ROI/stake +127.5%)
- Kelly kasa: **1000 → 1.03e+16 TL**

> ⚠️ **Backtest İYİMSER.** Aynı modelin canlı forward-testinde Ganyan ROI **~ −36%** çıktı (18 koşu). Geçmiş OOF backtest geleceği garanti ETMEZ; gerçek hakem forward-test'tir. Kelly'nin büyük görünmesi, dairesel-olmayan ama iyimser edge'in üst üste katlanmasıdır.


## 3) Egzotik bahis türleri — GÖSTERGE (literal TL DEĞİL)

> ⚠️ Egzotik geçmiş ödemesi yok → ödeme piyasa-ima ile tahmin edildi `(1−takeout)/P_market`. Skorlamada da aynı tahmin kullanıldığından bu **dairesel**; ROI fantazidir. Yalnız *hangi türde sinyal var* fikri için; bankroll olarak alınmaz. Gerçek değerlendirme için yukarıdaki (1) tanısına ve forward-test'e bakın.

| Tür | n | İsabet | Ort. ödeme~ | Ort. EV~ |
|-----|---|--------|-------------|----------|
| İkili | 3725 | 13.4% | 22.88 | +135.8% |
| Sıralı İkili | 2902 | 11.9% | 26.98 | +109.1% |
| Üçlü | 800 | 8.4% | 29.72 | +111.6% |
| Ganyan | 707 | 36.1% | 6.58 | +85.7% |
| Plase | 502 | 70.9% | 2.10 | +29.5% |
| Tabela | 267 | 7.1% | 30.61 | +110.0% |

### Çoklu-koşu (DENEYSEL — en yüksek varyans, gösterge)

| Tür | n | İsabet | Ort. ödeme~ | Ort. EV~ |
|-----|---|--------|-------------|----------|
| 3'lü Ganyan | 3457 | 10.6% | 86.02 | +183.8% |
| Çifte | 3173 | 22.0% | 20.02 | +100.8% |

---
*Araştırma/kâğıt-üzeri amaçlı. Gerçek bahis önerilmez.*