# Olasılık Kalibrasyonu — OOF (sızıntısız)

> Model %X diyorsa gerçekten ~%X mı oluyor? **Brier** (düşük=iyi), **ECE** (beklenen kalibrasyon hatası, düşük=iyi) ve reliability diagram ile ölçülür. Production modeller sınıf-dengeleme kullandığından olasılıklar sıralama için iyi ama kalibrasyon için bozuk (genelde **over-confident**) olabilir — bu, backtest'in neden forward-test'ten iyimser çıktığını kısmen açıklar.

**Kapsam:** 60,713 at-koşu kaydı (OOF).


## Kazanan (Is_Winner)

- **Brier (ham):** 0.0743  •  **ECE (ham):** 0.0075  •  taban oran: 10.0%
- **Brier (isotonic):** 0.0738  •  **ECE (isotonic):** 0.0000
- Yüksek-olasılık bölgesi (≥0.5, ham): **under-confident (tahmin < gerçek)**
- Grafik: `reports/calibration_is_winner.png`

| Bin | n | Ort. tahmin | Gözlenen |
|-----|---|-------------|----------|
| [0.0,0.1) | 42803 | 0.029 | 0.034 |
| [0.1,0.2) | 7154 | 0.144 | 0.146 |
| [0.2,0.3) | 4326 | 0.248 | 0.227 |
| [0.3,0.4) | 3310 | 0.348 | 0.327 |
| [0.4,0.5) | 2291 | 0.446 | 0.442 |
| [0.5,0.6) | 819 | 0.534 | 0.620 |
| [0.6,0.7) | 10 | 0.606 | 0.900 |
| [0.7,0.8) | 0 | — | — |
| [0.8,0.9) | 0 | — | — |
| [0.9,1.0) | 0 | — | — |

## İlk-3 (Is_Top3)

- **Brier (ham):** 0.1730  •  **ECE (ham):** 0.1340  •  taban oran: 30.1%
- **Brier (isotonic):** 0.1523  •  **ECE (isotonic):** 0.0000
- Yüksek-olasılık bölgesi (≥0.5, ham): **over-confident (tahmin > gerçek)**
- Grafik: `reports/calibration_is_top3.png`

| Bin | n | Ort. tahmin | Gözlenen |
|-----|---|-------------|----------|
| [0.0,0.1) | 5900 | 0.052 | 0.013 |
| [0.1,0.2) | 7738 | 0.152 | 0.058 |
| [0.2,0.3) | 7987 | 0.249 | 0.118 |
| [0.3,0.4) | 7620 | 0.350 | 0.185 |
| [0.4,0.5) | 6938 | 0.449 | 0.273 |
| [0.5,0.6) | 6379 | 0.549 | 0.371 |
| [0.6,0.7) | 6052 | 0.650 | 0.475 |
| [0.7,0.8) | 6143 | 0.749 | 0.584 |
| [0.8,0.9) | 4907 | 0.846 | 0.757 |
| [0.9,1.0) | 1049 | 0.921 | 0.929 |

---
> **Yorum:** Üretim zinciri (Stage 6 tahmin + Stage 8 EV) olasılıkları **isotonic** kalibratörden geçirir; yukarıdaki 'isotonic' satırı bu düzeltmenin OOF üzerindeki etkisidir. Isotonic monotondur → sıralama metrikleri (AUC, P@1) değişmez, yalnız EV gerçekçileşir.