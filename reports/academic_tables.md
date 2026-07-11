
### Tablo 1: Kazanan Tahmini İçin Model Performansları (Is_Winner)
| Model | AUC | Precision@1 | Value Strategy ROI |
|-------|-----|-------------|--------------------|
| CatBoost | 0.8452 | 41.8% | +19.3% |
| VotingEnsemble | 0.8417 | 41.2% | +17.0% |
| LightGBM | 0.8410 | 40.8% | +17.4% |
| XGBoost | 0.8404 | 41.1% | +18.6% |
| StackingEnsemble | 0.8391 | 40.6% | +94.3% |
| BaggingLGBM | 0.8387 | 40.4% | +16.6% |
| GradientBoosting | 0.8324 | 40.4% | +60.2% |
| LogisticRegression | 0.8243 | 38.6% | +8.6% |
| RandomForest | 0.8195 | 37.5% | -7.5% |

### Tablo 2: Tabela Tahmini İçin Model Performansları (Is_Top3)

> Not: Top3 için ROI raporlanmaz — veride yalnızca kazanma ganyanı bulunduğundan
> plase finişe parasal getiri hesaplamak yanıltıcı olur. Değerlendirme sıralama
> metrikleriyle yapılır.

| Model | AUC | Precision@1 | Precision@3 |
|-------|-----|-------------|-------------|
| StackingEnsemble | 0.8200 | 73.3% | 96.4% |
| LightGBM | 0.8196 | 72.8% | 96.5% |
| CatBoost | 0.8195 | 73.1% | 96.4% |
| VotingEnsemble | 0.8195 | 73.4% | 96.7% |
| XGBoost | 0.8194 | 73.0% | 96.7% |
| BaggingLGBM | 0.8183 | 72.9% | 96.6% |
| GradientBoosting | 0.8104 | 72.2% | 96.3% |
| LogisticRegression | 0.7983 | 71.0% | 96.1% |
| RandomForest | 0.7913 | 70.2% | 95.9% |
