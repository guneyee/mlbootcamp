# Home Credit Default Risk — Credit Scoring (FastAPI)

## Problem
- Amaç: Başvuru yapan müşterinin temerrüt riskini (TARGET) olasılık olarak tahmin etmek.
- Veri: Kaggle “Home Credit Default Risk”.
- Metrik: AUC (ek olarak business eşiği için precision/recall@k).

## Veri
- Kaynak: https://www.kaggle.com/competitions/home-credit-default-risk
- Ana tablo: `application_train.csv` (TARGET label’ı dahil).
- Ek tablolar (opsiyonel): bureau, bureau_balance, POS_CASH_balance, installments_payments, credit_card_balance.

## Kurulum
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Veri Yerleşimi
```
data/raw/application_train.csv
data/raw/application_test.csv   # opsiyonel
```
Ek tablolar varsa yine `data/raw/` altına.

## Pipeline Çalıştırma
```bash
python -m src.pipeline \
  --train-path data/raw/application_train.csv \
  --model-path models/model_lgbm.joblib \
  --preprocessor-path models/preprocessor.joblib \
  --feature-list-path models/feature_list.json
```

## FastAPI Servisi
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```
- `POST /predict`: Tekil veya batch JSON girişi.
- `GET /health`: Sağlık kontrolü.

Örnek:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"SK_ID_CURR":100002,"AMT_INCOME_TOTAL":202500,"AMT_CREDIT":406597.5,"NAME_CONTRACT_TYPE":"Cash loans"}'
```

## Notebooks
- `notebooks/01_eda.ipynb`
- `notebooks/02_baseline.ipynb`
- `notebooks/03_feature_eng.ipynb`
- `notebooks/04_model_opt.ipynb`
- `notebooks/05_eval.ipynb`
- `notebooks/06_pipeline.ipynb`

## Docs
- `docs/eda.md`
- `docs/baseline.md`
- `docs/feature_eng.md`
- `docs/model_opt.md`
- `docs/eval.md`
- `docs/final_pipeline.md`

## İzleme ve Canlıya Alım
- Metrikler: AUC, precision/recall@k, drift (PSI), veri kalitesi kontrolleri, latency.
- Eşik seçimi: Business gereksinimine göre (false negative/positive maliyetleri).

## Repo Yapısı (öneri)
```
data/
models/
notebooks/
docs/
src/
  config.py
  pipeline.py
  inference.py
  app.py
  utils/
tests/
```
