# Piyasa Sinyali Ablasyonu — Tam Model vs Ganyansız Model

> `Ganyan_*` (bahis oranı) özellikleri çıkarılarak modelin SADECE projenin kendi
> verileriyle (handikap, jokey/antrenör, soy hattı, idman) ne kadar başarılı olduğu
> ölçülmüştür. ΔAUC küçükse → projenin kendi feature'ları tek başına güçlüdür.


### Tablo A: Kazanan Tahmini (Is_Winner)

| Model | AUC (tam) | AUC (Ganyansız) | ΔAUC | P@1 (tam) | P@1 (Ganyansız) | ΔP@1 |
|-------|-----------|------------------|------|-----------|------------------|------|
| CatBoost | 0.8452 | 0.8218 | +0.0234 | 41.8% | 40.4% | +1.4% |
| VotingEnsemble | 0.8417 | 0.8199 | +0.0218 | 41.2% | 40.3% | +0.9% |
| LightGBM | 0.8410 | 0.8201 | +0.0209 | 40.8% | 40.3% | +0.6% |
| XGBoost | 0.8404 | 0.8189 | +0.0215 | 41.1% | 38.9% | +2.2% |
| StackingEnsemble | 0.8391 | 0.8183 | +0.0208 | 40.6% | 39.7% | +1.0% |
| BaggingLGBM | 0.8387 | 0.8163 | +0.0224 | 40.4% | 39.4% | +1.0% |
| GradientBoosting | 0.8324 | 0.8053 | +0.0271 | 40.4% | 37.6% | +2.7% |
| LogisticRegression | 0.8243 | 0.7949 | +0.0294 | 38.6% | 36.0% | +2.7% |
| RandomForest | 0.8195 | 0.7799 | +0.0396 | 37.5% | 34.3% | +3.3% |

*Ortalama ΔAUC (Is_Winner): +0.0252 (piyasa sinyalinin ortalama katkısı).*

### Tablo B: Tabela Tahmini (Is_Top3)

| Model | AUC (tam) | AUC (Ganyansız) | ΔAUC | P@1 (tam) | P@1 (Ganyansız) | ΔP@1 |
|-------|-----------|------------------|------|-----------|------------------|------|
| StackingEnsemble | 0.8200 | 0.8007 | +0.0193 | 73.3% | 69.9% | +3.4% |
| LightGBM | 0.8196 | 0.7995 | +0.0201 | 72.8% | 70.0% | +2.8% |
| CatBoost | 0.8195 | 0.8015 | +0.0180 | 73.1% | 70.8% | +2.3% |
| VotingEnsemble | 0.8195 | 0.7991 | +0.0204 | 73.4% | 70.5% | +2.9% |
| XGBoost | 0.8194 | 0.7992 | +0.0202 | 73.0% | 70.3% | +2.7% |
| BaggingLGBM | 0.8183 | 0.7982 | +0.0201 | 72.9% | 70.5% | +2.4% |
| GradientBoosting | 0.8104 | 0.7872 | +0.0232 | 72.2% | 69.5% | +2.7% |
| LogisticRegression | 0.7983 | 0.7735 | +0.0248 | 71.0% | 67.0% | +4.0% |
| RandomForest | 0.7913 | 0.7530 | +0.0383 | 70.2% | 66.3% | +3.8% |

*Ortalama ΔAUC (Is_Top3): +0.0227 (piyasa sinyalinin ortalama katkısı).*
