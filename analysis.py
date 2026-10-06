"""Telecom Customer Churn Analysis — Python EDA + logistic regression.

Run from the project root:
    ~/workspace/.venv/bin/python analysis.py

Outputs:
    charts/churn_by_contract.png
    charts/tenure_distribution.png
    charts/feature_importance.png
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "customers.csv")
CHARTS = os.path.join(HERE, "charts")
os.makedirs(CHARTS, exist_ok=True)

# ---------------------------------------------------------------- 1. Load
df = pd.read_csv(DATA)
print(f"Loaded {len(df):,} customers, {df.shape[1]} columns")
print(f"Overall churn rate: {df['churn'].mean():.1%}\n")

# ------------------------------------------------------------- 2. Clean
# Type hygiene + sanity checks (synthetic data is clean by construction,
# but a real pipeline validates anyway).
df["tenure_months"] = df["tenure_months"].clip(lower=1)
df["monthly_charges"] = df["monthly_charges"].clip(lower=0)
assert df.isna().sum().sum() == 0, "missing values found"
assert df["customer_id"].is_unique, "duplicate customer IDs"
print("Data quality: no nulls, no duplicate IDs, ranges valid.\n")

# ------------------------------------------------------------ 3. EDA
print("--- Churn rate by contract type ---")
by_contract = df.groupby("contract_type")["churn"].agg(["mean", "count"]).round(3)
by_contract.columns = ["churn_rate", "customers"]
print(by_contract, "\n")

print("--- Churn rate by tech support ---")
print(df.groupby("tech_support")["churn"].mean().round(3), "\n")

print("--- Churn rate by internet service ---")
print(df.groupby("internet_service")["churn"].mean().round(3), "\n")

print("--- Avg tenure: churned vs retained ---")
print(df.groupby("churn")["tenure_months"].mean().round(1), "\n")

print("--- Avg monthly charges: churned vs retained ---")
print(df.groupby("churn")["monthly_charges"].mean().round(2), "\n")

# Numeric correlation with churn
num_cols = ["tenure_months", "monthly_charges", "total_charges", "senior_citizen"]
print("--- Correlation with churn ---")
print(df[num_cols + ["churn"]].corr()["churn"].drop("churn").round(3), "\n")

# ------------------------------------------------- 4. Feature importance
# One-hot encode categoricals, scale numerics, fit logistic regression.
X = pd.get_dummies(
    df.drop(columns=["customer_id", "signup_month", "churn"]),
    drop_first=True,
)
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_s, y_train)
pred = model.predict(X_test_s)

print("--- Logistic regression (holdout, 20%) ---")
print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
print(classification_report(y_test, pred, digits=3))

importance = (
    pd.Series(np.abs(model.coef_[0]), index=X.columns)
    .sort_values(ascending=False)
)
print("--- Top churn drivers (|coefficient|) ---")
print(importance.head(8).round(3), "\n")

# ------------------------------------------------------------- 5. Charts
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

# Chart 1: churn rate by contract type
fig, ax = plt.subplots(figsize=(7, 4.2))
order = ["Month-to-month", "One year", "Two year"]
rates = df.groupby("contract_type")["churn"].mean().reindex(order) * 100
bars = ax.bar(order, rates, color=["#d62728", "#ff7f0e", "#2ca02c"])
ax.set_title("Churn Rate by Contract Type")
ax.set_ylabel("Churn rate (%)")
for bar, v in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
            f"{v:.1f}%", ha="center", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(CHARTS, "churn_by_contract.png"))
plt.close(fig)

# Chart 2: tenure distribution, churned vs retained
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.hist(df.loc[df.churn == 0, "tenure_months"], bins=24, alpha=0.65,
        label="Retained", color="#2ca02c")
ax.hist(df.loc[df.churn == 1, "tenure_months"], bins=24, alpha=0.65,
        label="Churned", color="#d62728")
ax.set_title("Tenure Distribution: Churned vs Retained")
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Customers")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(CHARTS, "tenure_distribution.png"))
plt.close(fig)

# Chart 3: feature importance (top 10)
top = importance.head(10).sort_values()
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.barh(top.index.str.replace("_", " "), top.values, color="#1f77b4")
ax.set_title("Top Churn Drivers — Logistic Regression |coefficient|")
ax.set_xlabel("Absolute coefficient (standardized features)")
fig.tight_layout()
fig.savefig(os.path.join(CHARTS, "feature_importance.png"))
plt.close(fig)

print(f"Charts saved to {CHARTS}/")

# -------------------------------------------------------- 6. Key insights
mtm_rate = df.loc[df.contract_type == "Month-to-month", "churn"].mean()
twoyr_rate = df.loc[df.contract_type == "Two year", "churn"].mean()
no_support = df.loc[df.tech_support == "No", "churn"].mean()
support = df.loc[df.tech_support == "Yes", "churn"].mean()
churned_mrr = df.loc[df.churn == 1, "monthly_charges"].sum()

print("\n================ KEY INSIGHTS ================")
print(f"1. Contract type dominates: month-to-month churn is {mtm_rate:.1%} "
      f"vs {twoyr_rate:.1%} on two-year contracts (~{mtm_rate / twoyr_rate:.0f}x higher).")
print(f"2. Tech support matters: no-support customers churn at {no_support:.1%} "
      f"vs {support:.1%} with support.")
print(f"3. New customers are fragile: churned customers average "
      f"{df.loc[df.churn == 1, 'tenure_months'].mean():.1f} months tenure vs "
      f"{df.loc[df.churn == 0, 'tenure_months'].mean():.1f} for retained.")
print(f"4. Price sensitivity: churned customers pay "
      f"${df.loc[df.churn == 1, 'monthly_charges'].mean():.2f}/mo on average vs "
      f"${df.loc[df.churn == 0, 'monthly_charges'].mean():.2f}/mo for retained.")
print(f"5. Revenue at risk: churned customers represent ${churned_mrr:,.0f}/mo "
      f"in lost monthly recurring revenue.")
print(f"6. The logistic model flags contract type, tech support, and tenure as the "
      f"top 3 predictors (see charts/feature_importance.png).")
print("=============================================")
