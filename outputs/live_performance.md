# Canlı Forward-Test Performansı (Kümülatif)

Eşleşen yarış sayısı: **147** | Tarih aralığı: 2026-07-12 → 2026-07-25

| Varyant | P@1 (winner) | ROI (winner, top1) | Bahis | P@1 (top3) | P@3 (top3) | Tabela 3/3 | Tabela 2/3+ |
|---------|--------------|--------------------|-------|------------|------------|------------|-------------|
| full | 27.9% | -38.9% | 146 | 62.6% | 95.2% | 8.2% | 63.3% |
| abl | 21.8% | -13.6% | 145 | 52.4% | 92.5% | 7.5% | 47.6% |
| blend | 34.5% | -16.0% | 110 | 62.7% | 95.5% | 10.0% | 64.5% |

> `full` = ganyanlı model, `abl` = ganyansız (erken) model. ROI yalnız kazanma bahsi içindir; final ganyanla ödenir. Kısa dönemde yüksek varyans normaldir — anlamlı sonuç için çok sayıda yarış gerekir.
