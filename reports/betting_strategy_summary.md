# Bahis Stratejisi — Backtest Özeti

**Kapsam:** 6,470 koşu (OOF, sızıntısız) • λ=0.85 • EV eşiği=+15%


## 1) Model vs Piyasa — seçim isabeti (ödeme VARSAYMADAN)

> En sağlam, dairesel-olmayan kanıt: modelin doğal seçimi (olasılığa göre ilk-k) piyasanın doğal seçiminden (favori sırası) daha sık mı tutuyor? Pozitif Δ → model o türde değer katıyor. **McNemar exact** testi (eşleşmiş ikili sonuç) farkın şans olup olmadığını ölçer: p<0.05 anlamlı.

| Bahis | n | Model isabet | Piyasa isabet | Δ (pp) | McNemar p |
|-------|---|--------------|---------------|--------|-----------|
| Ganyan | 6470 | 40.5% | 32.8% | +7.7 | 2.1e-39 ✓ |
| İkili | 6470 | 21.2% | 17.6% | +3.6 | 5.2e-13 ✓ |
| Sıralı İkili | 6470 | 12.6% | 9.6% | +3.0 | 7.2e-11 ✓ |
| Üçlü | 6470 | 4.3% | 3.2% | +1.2 | 7.0e-05 ✓ |
| Tabela | 6460 | 1.7% | 1.2% | +0.5 | 1.2e-02 ✓ |
| Plase | 6470 | 75.6% | 69.1% | +6.6 | 6.7e-37 ✓ |

## 2) Ganyan kasası — GERÇEK oran (tek güvenilir bankroll)

> Yalnız Ganyan'da gerçek ödeme (oran) var → bu kasa dairesel değil. Flat = sabit pay, Kelly = ¼-Kelly (cap %5).

- Bahis: **864** • isabet: **45.5%**
- Flat kasa: **1000 → 22190 TL** (ROI/stake +245.2%)
- Kelly kasa: **1000 → 1.69e+33 TL**

> ⚠️ **Backtest İYİMSER.** Aynı modelin canlı forward-testinde Ganyan ROI **~ −36%** çıktı (18 koşu). Geçmiş OOF backtest geleceği garanti ETMEZ; gerçek hakem forward-test'tir. Kelly'nin büyük görünmesi, dairesel-olmayan ama iyimser edge'in üst üste katlanmasıdır.


## 3) Egzotik bahis türleri — GÖSTERGE (literal TL DEĞİL)

> ⚠️ Egzotik geçmiş ödemesi yok → ödeme piyasa-ima ile tahmin edildi `(1−takeout)/P_market`. Skorlamada da aynı tahmin kullanıldığından bu **dairesel**; ROI fantazidir. Yalnız *hangi türde sinyal var* fikri için; bankroll olarak alınmaz. Gerçek değerlendirme için yukarıdaki (1) tanısına ve forward-test'e bakın.

| Tür | n | İsabet | Ort. ödeme~ | Ort. EV~ |
|-----|---|--------|-------------|----------|
| İkili | 3648 | 12.2% | 23.56 | +194.2% |
| Sıralı İkili | 2870 | 11.5% | 27.36 | +177.6% |
| Üçlü | 886 | 9.8% | 29.49 | +195.0% |
| Ganyan | 864 | 45.5% | 7.33 | +192.2% |
| Plase | 343 | 66.8% | 2.24 | +57.7% |
| Tabela | 294 | 8.2% | 30.15 | +195.7% |

### Çoklu-koşu (DENEYSEL — en yüksek varyans, gösterge)

| Tür | n | İsabet | Ort. ödeme~ | Ort. EV~ |
|-----|---|--------|-------------|----------|
| 3'lü Ganyan | 3183 | 16.0% | 116.91 | +1026.1% |
| Çifte | 2951 | 28.4% | 25.08 | +352.8% |

---
*Araştırma/kâğıt-üzeri amaçlı. Gerçek bahis önerilmez.*