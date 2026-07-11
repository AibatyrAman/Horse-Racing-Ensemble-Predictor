# Ganyancı İddiaları — İstatistiksel Test

**Kaynak:** `yaris_ana_tablo.csv` (73,036 at-koşu kaydı, 7,303 yarış). Cinsiyet parse oranı %100.0.

## 0) Genel: cinsiyet bazında kazanma oranları

| Cinsiyet | n | Kazanma | Tabela |
|----------|---|---------|--------|
| aygır (a) | 23,469 | %10.9 | %31.8 |
| dişi tay (d) | 17,657 | %8.0 | %26.2 |
| erkek tay (e) | 18,806 | %12.5 | %35.3 |
| iğdiş (g) | 905 | %7.7 | %29.8 |
| kısrak (k) | 12,199 | %7.5 | %24.2 |

## 1) İddia: erkek ağırlıklı yarışta azınlık dişi avantajlı mı?

Azınlık dişi = yarıştaki dişi oranı ≤ eşik VE at dişi. Şans beklentisi = Σ(1/yarış_at_sayısı)/n — her atın rastgele kazanma olasılığı.

| Eşik | n (azınlık dişi) | Kazanma | Şans beklentisi | Aynı yarış erkekleri | Binom p |
|------|------------------|---------|-----------------|----------------------|---------|
| ≤15% | 1,123 | %5.2 | %10.1 | %10.3 | 3.12e-09 |
| ≤25% | 3,369 | %4.7 | %10.0 | %10.8 | 3.1e-29 |
| ≤35% | 5,673 | %5.1 | %9.9 | %11.0 | 2.55e-39 |

Karma yarışlarda genel: dişi %5.8 (n=14,091) vs erkek %11.4 (n=32,322).

## 2) Bonus: don (tüy rengi) bazında kazanma oranı

| Don | n | Kazanma | Tabela |
|-----|---|---------|--------|
| al (a) | 22,437 | %9.9 | %29.1 |
| doru (d) | 33,048 | %10.3 | %31.2 |
| kır (k) | 17,550 | %9.6 | %29.2 |

> Don biyolojik bir performans belirleyici değildir; farklar büyük olasılıkla soy hattı karışıklığıdır (confounding). Modelde kullanılmaz, yalnız merak gideren yan bakış.

---
*Bu rapor tek-değişken (univariate) bakıştır; nihai hüküm, feature'ın çok-değişkenli modeldeki katkısıyla (ablation ΔAUC) verilir.*
