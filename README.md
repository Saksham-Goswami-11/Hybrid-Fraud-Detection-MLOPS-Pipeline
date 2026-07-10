# Fraud Detection System — Hybrid ML + Rule Engine

An end-to-end fraud detection system that combines a machine learning model with a rule-based risk engine, built to reflect how real fintech risk platforms actually operate — not just a notebook with a classifier at the end.

> **Why this project exists:** Fraud detection is a deceptively hard ML problem. With <1% of transactions being fraudulent, a model that predicts "not fraud" every single time is 99%+ "accurate" and completely useless. This project is built around that core challenge — and around the reality that no single model or rule catches everything.

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Key Insight: Why Accuracy Is the Wrong Metric](#key-insight-why-accuracy-is-the-wrong-metric)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [The Hybrid Risk Engine](#the-hybrid-risk-engine)
- [Results](#results)
- [Known Limitations & Edge Cases](#known-limitations--edge-cases)
- [Setup & Usage](#setup--usage)
- [Roadmap / Future Work](#roadmap--future-work)
- [What I Learned](#what-i-learned)

---

## Problem Statement

Financial institutions need to detect fraudulent transactions in real time while minimizing disruption to legitimate customers. This project builds a system that:

1. Scores every transaction with a fraud probability using a trained ML model
2. Runs an independent rule-based engine alongside the model to catch known high-risk patterns the model may miss
3. Explains every flagged prediction (SHAP), since fraud decisions in finance need to be auditable
4. Is deployed as a production-style API with versioning, CI/CD, and drift monitoring — not just a local script

**Dataset:** [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) (Kaggle, ULB Machine Learning Group) — ~285,000 anonymized transactions, ~0.17% labeled fraud.

---

## Key Insight: Why Accuracy Is the Wrong Metric

With this level of class imbalance, accuracy is misleading. This project optimizes for and reports:
- **Precision-Recall AUC** (not ROC-AUC, which overstates performance under imbalance)
- **Recall on the fraud class**, since missed fraud is typically far costlier than a false alarm
- **F2-score**, which explicitly weights recall over precision
- A rough **cost-based framing**: estimated cost of a missed fraud vs. cost of a false positive (analyst review time)

---

## Architecture

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
│   Demo UI   │     │   Endpoint   │     │   Container     │     │  + Rule     │
└─────────────┘     └──────┬───────┘     └────────────────┘     │  Engine     │
                            │                                    └─────────────┘
                            ▼
                   ┌─────────────────┐
                   │  Hybrid Decision │
                   │  ML score + Rules│
                   └─────────────────┘
       │
       ▼
┌─────────────────────┐
│  Monitoring/Drift    │
│  Dashboard           │
│  (Evidently)         │
└─────────────────────┘
```

**Design principle:** the ML model and the rule engine operate **independently**. A transaction can be flagged by either one — this is deliberate, since each layer is designed to catch what the other misses.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11, TypeScript |
| Data handling | pandas, numpy |
| Imbalance handling | imbalanced-learn (SMOTE, ADASYN) |
| Modeling | scikit-learn, XGBoost, LightGBM |
| Anomaly detection | Isolation Forest, Autoencoder (PyTorch) |
| Hyperparameter tuning | Optuna |
| Explainability | SHAP |
| Experiment tracking | MLflow |
| Data/model versioning | DVC |
| API | FastAPI + Pydantic |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Monitoring/drift detection | Evidently AI |
| Deployment | Render / AWS / GCP (free-tier) |
| Demo frontend | React (Vite, TypeScript, TailwindCSS) |

---

## Repository Structure

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
│   ├── config.py          # thresholds, rule parameters
│   ├── data_prep.py
│   ├── train.py
│   ├── evaluate.py
│   ├── rules.py            # rule-based risk engine
│   └── explain.py          # SHAP explainability
├── api/
│   ├── main.py              # FastAPI app
│   └── schemas.py           # Pydantic input/output models
├── frontend/                # React Dashboard client
│   ├── src/
│   │   ├── FraudDashboard.tsx
│   │   └── index.css
│   └── package.json
├── models/                  # saved model artifacts (DVC-tracked)
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

## The Hybrid Risk Engine

A pure ML threshold is not enough. This project uses three complementary strategies, matching how production fraud platforms (e.g., Stripe Radar, card-network risk engines) are typically built:

### 1. Classification Threshold (`src/config.py`)
The model's fraud probability is compared against a tunable threshold (default: `0.3104` optimized for F2-score). Lowering it increases recall (catches more fraud) at the cost of more false positives (larger analyst review queue). This trade-off is documented, not hidden — see [Results](#results).

### 2. Rule-Based Overrides
Independent of the ML score, rules trigger manual review or step-up authentication for known high-risk patterns.
*   **Rule 1: Escalation for high-amount transactions with moderate ML suspicion** (If amount > \$300.00 and ML fraud probability >= 10.0%, escalate to review).
*   **Rule 2 & 3: Behavioral Velocity Checks** (If card receives > 3 transactions or spends > \$1,000 within a rolling 10-minute window, trigger velocity override). *Note: Currently disabled/commented out for research.*

### 3. Retraining on False Negatives
Missed fraud cases are logged, reviewed, and used to retrain the model (`dvc repro`) with adjusted sample weights, so the model incrementally learns from what it previously missed.

---

## Results

Below are the final evaluation metrics computed on the hold-out test set:

| Model | Precision | Recall | F2-Score | PR-AUC | Cost ($) |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 48.78% | 80.00% | 70.92% | 0.7595 | 8,130.00 |
| **Random Forest (chosen best)** | **86.76%** | **78.67%** | **80.16%** | **0.8218** | **8,090.00** |
| LightGBM | 90.48% | 76.00% | 78.51% | 0.8025 | 9,060.00 |
| XGBoost | 89.06% | 76.00% | 78.30% | 0.7903 | 9,070.00 |
| Autoencoder (PyTorch) | 12.31% | 10.67% | 10.96% | 0.1202 | 34,070.00 |
| Isolation Forest | 7.24% | 52.00% | 23.24% | 0.0431 | 23,000.00 |

**Threshold trade-off (Random Forest):**
*   At threshold `0.3104`: precision = `86.76%`, recall = `78.67%` (9 false positives out of 68 flagged).
*   Lowering to `0.10`: increases recall to `85.33%` but increases the analyst review queue by **`158.8%`** (112 false positives out of 176 flagged).

**Estimated business impact:**
Based on an assumed average fraud loss of **\$500.00** per missed case and **\$10.00** analyst review cost per false positive, the final system (Random Forest at threshold `0.3104`) is estimated to minimize total business costs to **\$8,090.00** compared to the baseline Logistic Regression (\$8,130.00) on our test set.

---

## Known Limitations & Edge Cases

Documented honestly, because this is where the real engineering thinking shows:

*   **Low-amount fraud blind spot:** a transaction under the rule engine's amount threshold (e.g., under \$300) that also scores below the ML threshold (e.g., 11% vs. a 31% cutoff) will not be flagged by either layer. This is a known pattern for "card testing" fraud, where small transactions are used to validate a stolen card before a larger purchase. *Mitigation: a behavioral velocity rule (multiple small transactions in a short window) has been designed and tested.*
*   **Anonymized features (V1–V28):** the dataset's PCA-transformed features limit literal business interpretation of SHAP explanations — explanations are framed around relative importance and consistency rather than named business meaning.
*   **Static dataset:** the model is trained on a fixed historical dataset and does not reflect real-time evolving fraud patterns; in production this would require continuous retraining and monitoring.

---

## Setup & Usage

### Local setup
```bash
git clone https://github.com/<your-username>/fraud-detection-system.git
cd fraud-detection-system
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Reproduce the pipeline
```bash
venv/bin/dvc repro
```

### Run the API locally
```bash
uvicorn api.main:app --reload
```

### Run the Frontend client locally
```bash
cd frontend
npm install
npm run dev
```

### Example API request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"Time": 152802, "V1": -1.91, "V2": -0.49, "Amount": 357.95, "card_id": "card_demo"}'
```

### Example API response
```json
{
  "fraud_probability": 0.1156,
  "is_fraud": true,
  "threshold": 0.3104,
  "shap_values": {
    "V23": 0.1233,
    "V24": -0.1233
  },
  "top_risk_factors": [
    "V23 (+0.1233)",
    "V28 (+0.0886)"
  ],
  "rule_triggered": true,
  "rule_reasons": ["High Amount (> $300.00) with elevated ML risk (> 10.0%)"]
}
```

### Run tests
```bash
pytest tests/
```

---

## Roadmap / Future Work

- [ ] Enable/Enable-by-default the velocity-based rules and add geo-location coordinates
- [ ] Segment-specific thresholds (e.g., new accounts vs. established accounts)
- [ ] Streamlit/React dashboard for live drift monitoring
- [ ] A/B testing framework for comparing model versions in "shadow mode" before full rollout
- [ ] Expand test coverage across `src/`

---

## What I Learned

- Why accuracy is a misleading metric under severe class imbalance, and how to reason about precision/recall/F2 trade-offs in a business context
- Why production fraud systems are hybrid (ML + rules), not pure ML — and the specific blind spots that motivate that design
- End-to-end MLOps: experiment tracking, data/model versioning, containerization, CI/CD, and drift monitoring
- How to communicate model limitations and trade-offs honestly rather than only reporting favorable metrics

---

## License
MIT License