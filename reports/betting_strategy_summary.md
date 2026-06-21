# Bahis Stratejisi — Backtest Özeti

**Kapsam:** 1,830 koşu (OOF, sızıntısız) • λ=0.85 • EV eşiği=+5%


## 1) Model vs Piyasa — seçim isabeti (ödeme VARSAYMADAN)

> En sağlam, dairesel-olmayan kanıt: modelin doğal seçimi (olasılığa göre ilk-k) piyasanın doğal seçiminden (favori sırası) daha sık mı tutuyor? Pozitif Δ → model o türde değer katıyor. **McNemar exact** testi (eşleşmiş ikili sonuç) farkın şans olup olmadığını ölçer: p<0.05 anlamlı.

| Bahis | n | Model isabet | Piyasa isabet | Δ (pp) | McNemar p |
|-------|---|--------------|---------------|--------|-----------|
| Ganyan | 1830 | 39.8% | 31.7% | +8.1 | 1.6e-16 ✓ |
| İkili | 1830 | 22.1% | 16.4% | +5.7 | 1.1e-10 ✓ |
| Sıralı İkili | 1830 | 14.6% | 9.6% | +5.0 | 2.5e-10 ✓ |
| Üçlü | 1830 | 4.0% | 2.7% | +1.3 | 1.4e-02 ✓ |
| Tabela | 1830 | 1.6% | 0.9% | +0.7 | 4.1e-02 ✓ |
| Plase | 1830 | 73.0% | 68.3% | +4.7 | 1.9e-06 ✓ |

## 2) Ganyan kasası — GERÇEK oran (tek güvenilir bankroll)

> Yalnız Ganyan'da gerçek ödeme (oran) var → bu kasa dairesel değil. Flat = sabit pay, Kelly = ¼-Kelly (cap %5).

- Bahis: **178** • isabet: **39.9%**
- Flat kasa: **1000 → 3350 TL** (ROI/stake +132.1%)
- Kelly kasa: **1000 → 1.2e+06 TL**

> ⚠️ **Backtest İYİMSER.** Aynı modelin canlı forward-testinde Ganyan ROI **~ −36%** çıktı (18 koşu). Geçmiş OOF backtest geleceği garanti ETMEZ; gerçek hakem forward-test'tir. Kelly'nin büyük görünmesi, dairesel-olmayan ama iyimser edge'in üst üste katlanmasıdır.


## 3) Egzotik bahis türleri — GÖSTERGE (literal TL DEĞİL)

> ⚠️ Egzotik geçmiş ödemesi yok → ödeme piyasa-ima ile tahmin edildi `(1−takeout)/P_market`. Skorlamada da aynı tahmin kullanıldığından bu **dairesel**; ROI fantazidir. Yalnız *hangi türde sinyal var* fikri için; bankroll olarak alınmaz. Gerçek değerlendirme için yukarıdaki (1) tanısına ve forward-test'e bakın.

| Tür | n | İsabet | Ort. ödeme~ | Ort. EV~ |
|-----|---|--------|-------------|----------|
| İkili | 1018 | 16.0% | 22.73 | +142.5% |
| Sıralı İkili | 889 | 12.0% | 26.54 | +105.8% |
| Plase | 570 | 61.8% | 1.68 | +51.8% |
| Üçlü | 282 | 6.4% | 29.04 | +105.6% |
| Ganyan | 178 | 39.9% | 6.84 | +99.0% |
| Tabela | 97 | 5.2% | 31.64 | +91.6% |

### Çoklu-koşu (DENEYSEL — en yüksek varyans, gösterge)

| Tür | n | İsabet | Ort. ödeme~ | Ort. EV~ |
|-----|---|--------|-------------|----------|
| 3'lü Ganyan | 1058 | 7.0% | 76.35 | +138.5% |
| Çifte | 954 | 17.2% | 19.01 | +88.9% |

---
*Araştırma/kâğıt-üzeri amaçlı. Gerçek bahis önerilmez.*