# Piyasa Sinyali Ablasyonu — Tam Model vs Ganyansız Model

> `Ganyan_*` (bahis oranı) özellikleri çıkarılarak modelin SADECE projenin kendi
> verileriyle (handikap, jokey/antrenör, soy hattı, idman) ne kadar başarılı olduğu
> ölçülmüştür. ΔAUC küçükse → projenin kendi feature'ları tek başına güçlüdür.


### Tablo A: Kazanan Tahmini (Is_Winner)

| Model | AUC (tam) | AUC (Ganyansız) | ΔAUC | P@1 (tam) | P@1 (Ganyansız) | ΔP@1 |
|-------|-----------|------------------|------|-----------|------------------|------|
| CatBoost | 0.8412 | 0.8284 | +0.0128 | 39.1% | 38.0% | +1.1% |
| XGBoost | 0.8403 | 0.8297 | +0.0106 | 39.6% | 38.4% | +1.2% |
| StackingEnsemble | 0.8398 | 0.8267 | +0.0131 | 39.9% | 38.1% | +1.7% |
| LightGBM | 0.8393 | 0.8278 | +0.0115 | 38.5% | 38.3% | +0.2% |
| VotingEnsemble | 0.8377 | 0.8254 | +0.0123 | 38.4% | 36.6% | +1.8% |
| GradientBoosting | 0.8361 | 0.8231 | +0.0130 | 39.1% | 36.6% | +2.5% |
| LogisticRegression | 0.8332 | 0.8189 | +0.0143 | 38.7% | 36.2% | +2.6% |
| RandomForest | 0.8273 | 0.8030 | +0.0243 | 39.0% | 34.7% | +4.3% |
| BaggingLGBM | 0.8272 | 0.8158 | +0.0114 | 38.0% | 35.4% | +2.6% |

*Ortalama ΔAUC (Is_Winner): +0.0137 (piyasa sinyalinin ortalama katkısı).*

### Tablo B: Tabela Tahmini (Is_Top3)

| Model | AUC (tam) | AUC (Ganyansız) | ΔAUC | P@1 (tam) | P@1 (Ganyansız) | ΔP@1 |
|-------|-----------|------------------|------|-----------|------------------|------|
| XGBoost | 0.8158 | 0.7950 | +0.0208 | 73.2% | 69.6% | +3.7% |
| LightGBM | 0.8149 | 0.7940 | +0.0209 | 71.4% | 68.9% | +2.5% |
| StackingEnsemble | 0.8147 | 0.7929 | +0.0218 | 72.0% | 68.6% | +3.4% |
| CatBoost | 0.8146 | 0.7940 | +0.0206 | 73.0% | 69.3% | +3.7% |
| VotingEnsemble | 0.8142 | 0.7928 | +0.0214 | 71.3% | 68.5% | +2.8% |
| GradientBoosting | 0.8126 | 0.7929 | +0.0197 | 72.1% | 68.8% | +3.3% |
| BaggingLGBM | 0.8091 | 0.7886 | +0.0205 | 71.9% | 67.1% | +4.8% |
| LogisticRegression | 0.8073 | 0.7865 | +0.0208 | 70.9% | 67.4% | +3.4% |
| RandomForest | 0.7983 | 0.7652 | +0.0331 | 71.2% | 64.6% | +6.5% |

*Ortalama ΔAUC (Is_Top3): +0.0222 (piyasa sinyalinin ortalama katkısı).*
