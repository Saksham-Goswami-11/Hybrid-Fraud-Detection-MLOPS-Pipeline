# Product Requirements Document (PRD)
## End-to-End Fraud Detection System with MLOps

**Version:** 1.0
**Owner:** [Your Name]
**Status:** Draft
**Last Updated:** July 2026

---

## 1. Overview

### 1.1 Problem Statement
Financial institutions lose billions annually to fraudulent transactions. Fraud is extremely rare relative to legitimate transactions (often <1%), making it a hard machine learning problem: a naive model that predicts "not fraud" every time can still achieve >99% accuracy while catching zero fraud. This project builds an end-to-end system that detects fraudulent credit card transactions, explains *why* a transaction was flagged, and is deployed as a production-style API with monitoring — not just a notebook with a model.

### 1.2 Goals
- Build a fraud detection model that meaningfully outperforms a naive baseline on precision/recall, not just accuracy
- Make predictions explainable (regulatory and business necessity in fraud/finance)
- Package the model as a deployable API with proper MLOps practices (versioning, CI/CD, monitoring)
- Produce a portfolio-quality artifact: clean repo, documentation, architecture diagram, and a working demo

### 1.3 Non-Goals
- This is not a real-time streaming fraud system (e.g., no Kafka/Flink pipeline) — batch/API-based scoring only
- Not focused on beating academic SOTA benchmarks — focused on sound methodology and production practices
- No real financial data or PII — uses public anonymized dataset only

### 1.4 Success Metrics
| Metric | Target |
|---|---|
| Precision-Recall AUC | Meaningfully above baseline (report exact number after EDA) |
| Recall on fraud class | Prioritized over precision (missing fraud is costlier than false alarms) |
| Model explainability | SHAP values available for every prediction |
| API latency | < 200ms per prediction |
| Pipeline reproducibility | One-command retrain via versioned pipeline |
| Portfolio outcome | Deployed live demo + GitHub repo with clear README |

---

## 2. Users & Use Cases

### 2.1 Primary "User" (for framing purposes)
A fraud analyst or risk team at a financial institution who needs to:
- Get a fraud probability score per transaction
- Understand *why* a transaction was flagged (for compliance and manual review)
- Trust that the model is monitored for performance drift over time

### 2.2 Example Use Case Flow
1. Transaction data comes in (batch or single record via API)
2. Model scores the transaction with a fraud probability
3. If score exceeds threshold, transaction is flagged for review
4. Analyst views SHAP explanation for the flag
5. Ops team monitors model performance/drift over time via dashboard

---

## 3. Data

### 3.1 Dataset
- **Source:** Credit Card Fraud Detection dataset (Kaggle, ULB Machine Learning Group)
- **Size:** ~285,000 transactions, 492 labeled as fraud (~0.17%)
- **Features:** 28 anonymized PCA-transformed features (V1–V28) + `Time`, `Amount`, and `Class` (target)
- **Limitations:** Features are anonymized (no raw feature meaning), dataset is static (no live data feed), somewhat dated distribution of transaction patterns

### 3.2 Data Considerations
- Extreme class imbalance is the central challenge — must be addressed explicitly, not ignored
- No missing values in this dataset, but the pipeline should still include a data validation step (as if working with messier real-world data)
- Train/test split must be time-aware if possible (avoid leakage from future transactions into training)

---

## 4. Solution Design

### 4.1 High-Level Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐     ┌─────────────┐
│  Raw Data   │────▶│  Data Prep & │────▶│   Model         │────▶│  Model      │
│  (Kaggle)   │     │  Validation  │     │   Training      │     │  Registry   │
└─────────────┘     └──────────────┘     │  (MLflow logged)│     │  (MLflow)   │
                                          └────────────────┘     └──────┬──────┘
                                                                        │
                                                                        ▼
┌─────────────┐     ┌──────────────┐     ┌────────────────┐     ┌─────────────┐
│   Client /  │◀────│   FastAPI    │◀────│   Docker        │◀────│  Best Model │
│   Demo UI   │     │   Endpoint   │     │   Container     │     │  Loaded     │
└─────────────┘     └──────────────┘     └────────────────┘     └─────────────┘
       │
       ▼
┌─────────────────────┐
│  Monitoring/Drift    │
│  Dashboard           │
│  (Evidently)         │
└─────────────────────┘
```

### 4.2 Modeling Approach
| Approach | Why Include It |
|---|---|
| Logistic Regression (class-weighted) | Simple, interpretable baseline |
| Random Forest | Non-linear baseline, decent default performance |
| XGBoost / LightGBM (tuned with Optuna) | Best expected supervised performance |
| Isolation Forest | Unsupervised anomaly detection angle — useful when labels are scarce/delayed in real world |
| Autoencoder (Keras/PyTorch) | Deep learning anomaly detection via reconstruction error |
| Ensemble of best supervised + unsupervised | Final production candidate |

### 4.3 Imbalance Handling Strategies to Compare
- Class weighting
- SMOTE / ADASYN oversampling
- Random undersampling
- Threshold tuning based on cost matrix (false negative cost >> false positive cost)

### 4.4 Evaluation Strategy
- **Primary metric:** Precision-Recall AUC (not ROC-AUC, which is misleading under extreme imbalance)
- **Secondary metrics:** Recall @ fixed precision, F2-score (weights recall higher), confusion matrix at chosen threshold
- **Business framing:** Define an approximate cost matrix (e.g., cost of missed fraud vs. cost of false alarm) and report expected cost savings vs. baseline

### 4.5 Explainability
- SHAP (TreeExplainer for tree models, KernelExplainer or DeepExplainer for others)
- Global feature importance plot
- Local force plot for individual flagged transactions
- Expose SHAP values through the API response, not just in the notebook

---

## 5. Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| Data handling | pandas, numpy |
| Imbalance handling | imbalanced-learn (SMOTE, ADASYN) |
| Modeling | scikit-learn, XGBoost, LightGBM, PyTorch or TensorFlow/Keras (autoencoder) |
| Hyperparameter tuning | Optuna |
| Explainability | SHAP |
| Experiment tracking | MLflow |
| Data/model versioning | DVC |
| API | FastAPI + Pydantic (input validation) |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Monitoring/drift detection | Evidently AI |
| Deployment | Render / AWS (EC2 or ECS) / GCP Cloud Run — pick one free/low-cost option |
| Demo frontend (optional) | Streamlit (simple UI hitting the FastAPI endpoint) |
| Visualization | matplotlib, seaborn, plotly |
| Version control | Git + GitHub |

---

## 6. Roadmap

### Phase 1 — Foundation (Days 1–3)
- Set up repo structure, virtual environment, dependency management
- Load dataset, write data validation checks
- EDA: class distribution, feature distributions, correlation analysis
- Document 3–5 key insights

### Phase 2 — Baseline Modeling (Days 4–6)
- Train/test split (time-aware)
- Baseline: Logistic Regression with class weighting
- Establish evaluation harness (PR-AUC, F2, confusion matrix) — reused for all future models
- Log baseline run in MLflow

### Phase 3 — Advanced Modeling (Days 7–11)
- Try SMOTE/ADASYN/undersampling variants
- Train Random Forest, XGBoost, LightGBM
- Hyperparameter tuning with Optuna on best candidate
- Train Isolation Forest and Autoencoder as unsupervised comparisons
- Log all runs in MLflow, compare in a single leaderboard table

### Phase 4 — Explainability (Days 12–13)
- SHAP global + local explanations for the best model
- Sanity-check explanations against domain intuition (do flagged transactions make sense given feature values?)

### Phase 5 — MLOps Pipeline (Days 14–19)
- Set up DVC for data/model versioning
- Build FastAPI service wrapping the final model + SHAP explanation output
- Write Dockerfile, containerize the API
- Set up GitHub Actions: run tests + linting on every push
- Simulate data drift (e.g., inject shifted distributions) and build an Evidently report/dashboard

### Phase 6 — Deployment & Demo (Days 20–22)
- Deploy API to Render/AWS/GCP
- (Optional) Build simple Streamlit frontend hitting the deployed API
- Record a short demo (GIF or video) of a transaction being scored and explained

### Phase 7 — Portfolio Packaging (Days 23–25)
- Write comprehensive README: problem, why accuracy is misleading, architecture diagram, results table, trade-offs considered, how to run locally
- Add architecture diagram (the one above, polished)
- Clean up notebooks vs. production code separation (`/notebooks` vs `/src`)
- Push final repo, write a short LinkedIn/portfolio post summarizing the project and what you learned

**Total estimated timeline:** ~3.5–4 weeks at a steady, part-time pace (longer if MLOps tools are new to you — budget extra time for Docker/CI-CD if so)

---

## 7. Repository Structure (Suggested)

```
fraud-detection-system/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_model.ipynb
│   └── 03_advanced_modeling.ipynb
├── src/
│   ├── data_prep.py
│   ├── train.py
│   ├── evaluate.py
│   └── explain.py
├── api/
│   ├── main.py          # FastAPI app
│   └── schemas.py        # Pydantic input/output models
├── models/                # saved model artifacts (or DVC-tracked)
├── monitoring/
│   └── drift_report.py
├── tests/
│   └── test_api.py
├── .github/workflows/
│   └── ci.yml
├── Dockerfile
├── dvc.yaml
├── requirements.txt
├── README.md
└── architecture_diagram.png
```

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Overfitting to rare fraud patterns in a static dataset | Use proper cross-validation stratified by class; don't over-tune on test set |
| Getting stuck on MLOps tooling (Docker/CI-CD) as a beginner | Timebox each tool to 1–2 days; use official quickstart docs; skip advanced monitoring if time-constrained and note it as "future work" |
| Anonymized features limit interpretability of SHAP results | Frame explainability around relative feature importance and consistency, not literal business meaning |
| Free-tier deployment limits (cold starts, memory caps) | Choose Render or Railway for simplicity; document limitation in README |
| Scope creep (trying to add too many models/tools) | Lock scope after Phase 3; treat anything beyond as "stretch goals" in README |

---

## 9. Stretch Goals (Optional, if time allows)
- Add a simple rule-based fraud detector as an additional baseline for comparison
- Build a basic Streamlit dashboard showing live drift metrics
- Add A/B testing simulation between two model versions
- Write unit tests covering >80% of `src/` code

---

## 10. Definition of Done
- [ ] EDA complete with documented insights
- [ ] At least 4 models trained and logged in MLflow with a comparison table
- [ ] Best model selected based on PR-AUC and business cost framing
- [ ] SHAP explainability integrated into API response
- [ ] API containerized and passing CI checks
- [ ] Model deployed and publicly accessible via URL
- [ ] README complete with architecture diagram, results, and setup instructions
- [ ] Repo pushed to GitHub with clean commit history