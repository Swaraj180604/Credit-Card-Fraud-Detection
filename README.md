# Credit Card Fraud Detection

A machine learning pipeline for detecting fraudulent credit card transactions using a Random Forest classifier trained on synthetic transaction data.

---

## Overview

This project generates synthetic credit card transaction data, engineers meaningful fraud-detection features, trains a Random Forest model, evaluates its performance, and saves the model artifacts for downstream use.

---

## Model Performance

| Metric | Score |
|---|---|
| Accuracy | 1.00 |
| ROC-AUC | 1.0000 |
| Average Precision | 1.0000 |
| Fraud Precision | 1.00 |
| Fraud Recall | 1.00 |

> **Note:** Perfect scores are expected here because the data is **synthetically generated** with clearly separable fraud/legitimate distributions. Real-world performance will differ significantly.

---

## Project Structure

```
├── Credit_Card_Fraud_Detection.ipynb   # Main notebook (training + evaluation)
├── fraud_model.joblib                  # Trained Random Forest model
├── scaler.joblib                       # Fitted StandardScaler
├── feature_names.joblib                # Ordered list of feature names
└── README.md
```

---

## Dataset

Synthetic data with **50,000 transactions** at a **2% fraud rate** (1,000 fraud, 49,000 legitimate).

### Raw Features

| Feature | Description |
|---|---|
| `amount` | Transaction amount |
| `hour_of_day` | Hour the transaction occurred |
| `day_of_week` | Day of the week (0–6) |
| `merchant_category` | Merchant category code (0–9) |
| `num_transactions_1h` | Number of transactions in the past hour |
| `num_transactions_24h` | Number of transactions in the past 24 hours |
| `avg_amount_30d` | Average transaction amount over the past 30 days |
| `distance_from_home` | Distance of transaction location from home (km) |
| `is_online` | Whether the transaction was online (0/1) |
| `is_international` | Whether the transaction was international (0/1) |
| `card_present` | Whether the physical card was present (0/1) |
| `days_since_last_txn` | Days since the cardholder's last transaction |
| `credit_limit_used_pct` | Fraction of credit limit used |
| `velocity_score` | Pre-computed transaction velocity score |
| `geo_risk_score` | Pre-computed geographic risk score |

### Derived Features

| Feature | Formula |
|---|---|
| `amount_to_avg_ratio` | `amount / (avg_amount_30d + 1)` |
| `txn_burst` | `num_transactions_1h / (num_transactions_24h + 1)` |
| `risk_composite` | `(velocity_score + geo_risk_score) / 2` |

### Top Feature Importances

| Feature | Importance |
|---|---|
| `risk_composite` | 0.3348 |
| `geo_risk_score` | 0.1960 |
| `velocity_score` | 0.1504 |
| `credit_limit_used_pct` | 0.1262 |
| `num_transactions_24h` | 0.0706 |

---

## Model Details

- **Algorithm:** Random Forest Classifier (`sklearn`)
- **Estimators:** 200 trees
- **Max Depth:** 12
- **Class Weight:** `balanced` (handles class imbalance)
- **Train/Test Split:** 80/20, stratified
- **Feature Scaling:** `StandardScaler`

---

## Requirements

```
numpy
pandas
scikit-learn
joblib
```

Install with:

```bash
pip install numpy pandas scikit-learn joblib
```

---

## Usage

### 1. Train the model

Open and run the notebook end-to-end:

```bash
jupyter notebook Credit_Card_Fraud_Detection.ipynb
```

This will produce `fraud_model.joblib`, `scaler.joblib`, and `feature_names.joblib`.

### 2. Load and use the saved model

```python
import joblib
import pandas as pd

model         = joblib.load('fraud_model.joblib')
scaler        = joblib.load('scaler.joblib')
feature_names = joblib.load('feature_names.joblib')

# Prepare a new transaction as a DataFrame
transaction = pd.DataFrame([{
    'amount': 1500.0,
    'hour_of_day': 2,
    'day_of_week': 5,
    'merchant_category': 3,
    'num_transactions_1h': 5,
    'num_transactions_24h': 14,
    'avg_amount_30d': 120.0,
    'distance_from_home': 350.0,
    'is_online': 1,
    'is_international': 1,
    'card_present': 0,
    'days_since_last_txn': 0.2,
    'credit_limit_used_pct': 0.92,
    'velocity_score': 0.85,
    'geo_risk_score': 0.78,
    'amount_to_avg_ratio': 1500.0 / (120.0 + 1),
    'txn_burst': 5 / (14 + 1),
    'risk_composite': (0.85 + 0.78) / 2,
}])

X_scaled = scaler.transform(transaction[feature_names])
prediction   = model.predict(X_scaled)[0]
fraud_prob   = model.predict_proba(X_scaled)[0][1]

print(f"Prediction : {'FRAUD' if prediction == 1 else 'Legitimate'}")
print(f"Fraud prob : {fraud_prob:.4f}")
```

---

## Fraud Patterns (Synthetic Data Logic)

The synthetic fraud transactions are designed with the following characteristics vs. legitimate ones:

| Signal | Legitimate | Fraud |
|---|---|---|
| Amount | Lower (lognormal μ=4.0) | Higher (lognormal μ=5.5) |
| Hour | Business hours (6–22) | Late night / early morning (0–5, 22–23) |
| Transactions/hour | ~1.5 | ~4.5 |
| Distance from home | ~20 km | ~200 km |
| International | 5% | 50% |
| Credit limit used | Low (beta 2,8) | High (beta 8,2) |
| Velocity / geo risk | Low | High |

---

## Limitations

- This model is trained on **synthetic data** — it is not suitable for production use without retraining on real-world transaction data.
- Perfect evaluation metrics are an artifact of the synthetic data's clearly separable distributions.
- Real fraud detection requires additional considerations such as concept drift, evolving fraud patterns, and regulatory compliance.
