# End-to-End Fraud Detection System with MLOps

A production-grade fraud detection system that goes beyond notebook modeling — featuring explainable predictions, a deployable API, CI/CD, model versioning, and drift monitoring.

## 🎯 Problem

Financial institutions lose billions to fraud annually. Fraud is extremely rare (~0.17% of transactions), making it a deceptively hard ML problem: a naive "predict no fraud" model gets >99% accuracy while catching zero fraud. This project builds a system that **meaningfully detects fraud, explains why**, and is **deployed with proper MLOps practices**.

## 🏗️ Architecture

```
Raw Data → Data Validation → Model Training (MLflow tracked) → Model Registry
                                                                      ↓
Client / Demo UI ← FastAPI Endpoint ← Docker Container ← Best Model Loaded
       ↓
Monitoring / Drift Dashboard (Evidently)
```

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| Data | pandas, numpy |
| Modeling | scikit-learn, XGBoost, LightGBM, PyTorch (autoencoder) |
| Imbalance | imbalanced-learn (SMOTE, ADASYN) |
| Tuning | Optuna |
| Explainability | SHAP |
| Tracking | MLflow |
| Versioning | DVC |
| API | FastAPI + Pydantic |
| Container | Docker |
| CI/CD | GitHub Actions |
| Monitoring | Evidently AI |
| Deployment | Render |

## 🚀 Quick Start

```bash
# 1. Clone and set up environment
git clone <repo-url>
cd fraud-detection-system
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Download data (requires Kaggle API key)
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/ --unzip

# 3. Run the full pipeline
dvc repro

# 4. Start the API
uvicorn api.main:app --reload

# 5. View MLflow dashboard
mlflow ui
```

## 📊 Results

The models were trained and evaluated on 227K training samples and 57K test samples. 

| Model | PR-AUC (Primary) | ROC-AUC | F2-Score | Recall | Precision | Business Cost ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest 🏆** | **0.8218** | **0.9564** | **0.8016** | **78.67%** | **86.76%** | **$8,090.00** |
| LightGBM | 0.8025 | 0.9876 | 0.7851 | 76.00% | 90.48% | $9,060.00 |
| XGBoost | 0.7903 | 0.9818 | 0.7830 | 76.00% | 89.06% | $9,070.00 |
| Logistic Regression | 0.7595 | 0.9870 | 0.7092 | 80.00% | 48.78% | $8,130.00 |
| Autoencoder | 0.1202 | 0.9262 | 0.1096 | 10.67% | 12.31% | $34,070.00 |
| Isolation Forest | 0.0431 | 0.9489 | 0.2324 | 52.00% | 7.24% | $23,000.00 |

*   **Best Model**: `RandomForestClassifier` was selected (maximizing PR-AUC while maintaining high precision/recall balance). The decision threshold was optimized to `0.3104` using the cost matrix ($500 per missed fraud vs. $10 per false alarm).
*   **Deep Learning / Anomaly Detection**: Unsupervised methods like the PyTorch Autoencoder (PR-AUC: 0.1202) perform worse standalone but serve as good anomaly signals for ensembling.
*

## 📁 Project Structure

```
├── data/raw/              # Raw dataset (DVC-tracked)
├── data/processed/        # Train/test splits (DVC-tracked)
├── notebooks/             # EDA and modeling notebooks
├── src/                   # Production source code
├── api/                   # FastAPI service
├── models/                # Saved model artifacts
├── monitoring/            # Drift detection
├── tests/                 # Unit tests
├── .github/workflows/     # CI/CD
├── Dockerfile
├── dvc.yaml
└── requirements.txt
```

## 📝 License

MIT
