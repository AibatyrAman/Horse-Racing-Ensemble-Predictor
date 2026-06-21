# Olasılık Kalibrasyonu — OOF (sızıntısız)

> Model %X diyorsa gerçekten ~%X mı oluyor? **Brier** (düşük=iyi), **ECE** (beklenen kalibrasyon hatası, düşük=iyi) ve reliability diagram ile ölçülür. Production modeller sınıf-dengeleme kullandığından olasılıklar sıralama için iyi ama kalibrasyon için bozuk (genelde **over-confident**) olabilir — bu, backtest'in neden forward-test'ten iyimser çıktığını kısmen açıklar.

**Kapsam:** 17,923 at-koşu kaydı (OOF).


## Kazanan (Is_Winner)

- **Brier:** 0.0761  •  **ECE:** 0.0054  •  taban oran: 10.2%
- Yüksek-olasılık bölgesi (≥0.5): **under-confident (tahmin < gerçek)**
- Grafik: `reports/calibration_is_winner.png`

| Bin | n | Ort. tahmin | Gözlenen |
|-----|---|-------------|----------|
| [0.0,0.1) | 12204 | 0.033 | 0.032 |
| [0.1,0.2) | 2409 | 0.143 | 0.148 |
| [0.2,0.3) | 1360 | 0.246 | 0.235 |
| [0.3,0.4) | 990 | 0.347 | 0.314 |
| [0.4,0.5) | 646 | 0.445 | 0.415 |
| [0.5,0.6) | 297 | 0.536 | 0.569 |
| [0.6,0.7) | 17 | 0.614 | 0.941 |
| [0.7,0.8) | 0 | — | — |
| [0.8,0.9) | 0 | — | — |
| [0.9,1.0) | 0 | — | — |

## İlk-3 (Is_Top3)

- **Brier:** 0.2630  •  **ECE:** 0.3018  •  taban oran: 30.7%
- Yüksek-olasılık bölgesi (≥0.5): **over-confident (tahmin > gerçek)**
- Grafik: `reports/calibration_is_top3.png`

| Bin | n | Ort. tahmin | Gözlenen |
|-----|---|-------------|----------|
| [0.0,0.1) | 1507 | 0.053 | 0.009 |
| [0.1,0.2) | 1026 | 0.148 | 0.032 |
| [0.2,0.3) | 933 | 0.249 | 0.063 |
| [0.3,0.4) | 980 | 0.350 | 0.089 |
| [0.4,0.5) | 1278 | 0.451 | 0.128 |
| [0.5,0.6) | 1502 | 0.552 | 0.160 |
| [0.6,0.7) | 2051 | 0.652 | 0.224 |
| [0.7,0.8) | 2494 | 0.751 | 0.329 |
| [0.8,0.9) | 3234 | 0.854 | 0.479 |
| [0.9,1.0) | 2918 | 0.935 | 0.712 |

---
> **Yorum:** ECE büyük / over-confident ise, bahis EV hesabı için olasılıkları **isotonic veya Platt** ile yeniden kalibre etmek bir sonraki adım olabilir (sıralama metrikleri değişmez, EV gerçekçileşir). Bu rapor yalnız ölçer; düzeltme uygulamaz.