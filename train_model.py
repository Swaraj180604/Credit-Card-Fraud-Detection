"""
Credit Card Fraud Detection - Model Training Script
Run this file first to generate: fraud_model.joblib, scaler.joblib, feature_names.joblib
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score)
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. GENERATE SYNTHETIC CREDIT CARD DATA
# ─────────────────────────────────────────────
np.random.seed(42)
N = 50_000
FRAUD_RATE = 0.02  # 2% fraud

n_fraud = int(N * FRAUD_RATE)
n_legit = N - n_fraud

def make_legit(n):
    return pd.DataFrame({
        'amount':            np.random.lognormal(4.0, 1.2, n),
        'hour_of_day':       np.random.choice(range(6, 23), n),
        'day_of_week':       np.random.choice(range(7), n),
        'merchant_category': np.random.choice(range(10), n),
        'num_transactions_1h':  np.random.poisson(1.5, n),
        'num_transactions_24h': np.random.poisson(5, n),
        'avg_amount_30d':    np.random.lognormal(3.8, 0.9, n),
        'distance_from_home': np.random.exponential(20, n),
        'is_online':         np.random.choice([0, 1], n, p=[0.6, 0.4]),
        'is_international':  np.random.choice([0, 1], n, p=[0.95, 0.05]),
        'card_present':      np.random.choice([0, 1], n, p=[0.3, 0.7]),
        'days_since_last_txn': np.random.exponential(2, n),
        'credit_limit_used_pct': np.random.beta(2, 8, n),
        'velocity_score':    np.random.beta(2, 10, n),
        'geo_risk_score':    np.random.beta(1, 15, n),
    })

def make_fraud(n):
    return pd.DataFrame({
        'amount':            np.random.lognormal(5.5, 1.5, n),   # higher amounts
        'hour_of_day':       np.random.choice(list(range(0, 6)) + list(range(22, 24)), n),  # odd hours
        'day_of_week':       np.random.choice(range(7), n),
        'merchant_category': np.random.choice(range(10), n),
        'num_transactions_1h':  np.random.poisson(4.5, n),       # many transactions
        'num_transactions_24h': np.random.poisson(12, n),
        'avg_amount_30d':    np.random.lognormal(3.2, 0.8, n),   # amount deviation
        'distance_from_home': np.random.exponential(200, n),     # far from home
        'is_online':         np.random.choice([0, 1], n, p=[0.3, 0.7]),
        'is_international':  np.random.choice([0, 1], n, p=[0.5, 0.5]),  # often international
        'card_present':      np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'days_since_last_txn': np.random.exponential(0.5, n),    # very recent
        'credit_limit_used_pct': np.random.beta(8, 2, n),        # near limit
        'velocity_score':    np.random.beta(8, 2, n),
        'geo_risk_score':    np.random.beta(8, 2, n),
    })

df_legit = make_legit(n_legit)
df_legit['is_fraud'] = 0
df_fraud = make_fraud(n_fraud)
df_fraud['is_fraud'] = 1

df = pd.concat([df_legit, df_fraud]).sample(frac=1, random_state=42).reset_index(drop=True)

# Add derived features
df['amount_to_avg_ratio'] = df['amount'] / (df['avg_amount_30d'] + 1)
df['txn_burst']           = df['num_transactions_1h'] / (df['num_transactions_24h'] + 1)
df['risk_composite']      = (df['velocity_score'] + df['geo_risk_score']) / 2

FEATURES = [
    'amount', 'hour_of_day', 'day_of_week', 'merchant_category',
    'num_transactions_1h', 'num_transactions_24h', 'avg_amount_30d',
    'distance_from_home', 'is_online', 'is_international', 'card_present',
    'days_since_last_txn', 'credit_limit_used_pct', 'velocity_score',
    'geo_risk_score', 'amount_to_avg_ratio', 'txn_burst', 'risk_composite'
]

X = df[FEATURES]
y = df['is_fraud']

# ─────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# 3. SCALE FEATURES
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 4. TRAIN RANDOM FOREST (main model)
# ─────────────────────────────────────────────
print("Training Random Forest …")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_sc, y_train)

# ─────────────────────────────────────────────
# 5. EVALUATE
# ─────────────────────────────────────────────
y_pred  = model.predict(X_test_sc)
y_proba = model.predict_proba(X_test_sc)[:, 1]

print("\n" + "="*55)
print("  CREDIT CARD FRAUD DETECTION — MODEL EVALUATION")
print("="*55)
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
print(f"ROC-AUC Score      : {roc_auc_score(y_test, y_proba):.4f}")
print(f"Avg Precision Score: {average_precision_score(y_test, y_proba):.4f}")
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
print(f"  FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")

# Feature importances
fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nTop 10 Feature Importances:")
print(fi.head(10).to_string())

# ─────────────────────────────────────────────
# 6. SAVE ARTIFACTS
# ─────────────────────────────────────────────
joblib.dump(model,    'fraud_model.joblib')
joblib.dump(scaler,   'scaler.joblib')
joblib.dump(FEATURES, 'feature_names.joblib')

print("\n✅  Saved: fraud_model.joblib | scaler.joblib | feature_names.joblib")
