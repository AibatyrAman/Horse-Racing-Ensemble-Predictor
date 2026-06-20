
### Tablo 1: Kazanan Tahmini İçin Model Performansları (Is_Winner)
| Model | AUC | Precision@1 | Value Strategy ROI |
|-------|-----|-------------|--------------------|
| CatBoost | 0.8412 | 39.1% | +11.1% |
| XGBoost | 0.8403 | 39.6% | +30.0% |
| StackingEnsemble | 0.8398 | 39.9% | +86.8% |
| LightGBM | 0.8393 | 38.5% | +14.5% |
| VotingEnsemble | 0.8377 | 38.4% | +13.4% |
| GradientBoosting | 0.8361 | 39.1% | +88.3% |
| LogisticRegression | 0.8332 | 38.7% | +24.7% |
| RandomForest | 0.8273 | 39.0% | +3.4% |
| BaggingLGBM | 0.8272 | 38.0% | +12.2% |

### Tablo 2: Tabela Tahmini İçin Model Performansları (Is_Top3)

> Not: Top3 için ROI raporlanmaz — veride yalnızca kazanma ganyanı bulunduğundan
> plase finişe parasal getiri hesaplamak yanıltıcı olur. Değerlendirme sıralama
> metrikleriyle yapılır.

| Model | AUC | Precision@1 | Precision@3 |
|-------|-----|-------------|-------------|
| XGBoost | 0.8158 | 73.2% | 96.8% |
| LightGBM | 0.8149 | 71.4% | 96.9% |
| StackingEnsemble | 0.8147 | 72.0% | 96.9% |
| CatBoost | 0.8146 | 73.0% | 96.8% |
| VotingEnsemble | 0.8142 | 71.3% | 96.9% |
| GradientBoosting | 0.8126 | 72.1% | 97.0% |
| BaggingLGBM | 0.8091 | 71.9% | 96.6% |
| LogisticRegression | 0.8073 | 70.9% | 96.7% |
| RandomForest | 0.7983 | 71.2% | 96.5% |
