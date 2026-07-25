# Canlı Forward-Test Performansı (Kümülatif)

Eşleşen yarış sayısı: **144** | Tarih aralığı: 2026-07-12 → 2026-07-24

| Varyant | P@1 (winner) | ROI (winner, top1) | Bahis | P@1 (top3) | P@3 (top3) |
|---------|--------------|--------------------|-------|------------|------------|
| full | 27.8% | -39.8% | 143 | 61.8% | 95.1% |
| abl | 20.8% | -24.8% | 142 | 52.1% | 92.4% |
| blend | 34.6% | -16.5% | 107 | 61.7% | 95.3% |

> `full` = ganyanlı model, `abl` = ganyansız (erken) model. ROI yalnız kazanma bahsi içindir; final ganyanla ödenir. Kısa dönemde yüksek varyans normaldir — anlamlı sonuç için çok sayıda yarış gerekir.
