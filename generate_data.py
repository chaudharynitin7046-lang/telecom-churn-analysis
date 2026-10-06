"""Generate a deterministic synthetic telecom churn dataset.

Run: ~/workspace/.venv/bin/python generate_data.py
Output: data/customers.csv (8000 rows, seed=42)

Churn is modeled on realistic drivers:
  - month-to-month contracts churn far more than 1/2-year contracts
  - no tech support -> higher churn
  - low tenure -> higher churn
  - high monthly charges -> higher churn
  - senior citizens slightly higher churn
"""

import numpy as np
import pandas as pd

SEED = 42
N = 8000
OUT = "data/customers.csv"

rng = np.random.default_rng(SEED)

# ---- Customer IDs -----------------------------------------------------------
customer_id = [f"C{100000 + i}" for i in range(N)]

# ---- Tenure (months 1-72, skewed toward lower values) ------------------------
tenure_months = np.clip(rng.gamma(shape=2.0, scale=9.0, size=N).astype(int) + 1, 1, 72)

# ---- Contract type (correlated with tenure: new customers mostly monthly) ----
contract_type = []
for t in tenure_months:
    if t <= 12:
        p = [0.75, 0.15, 0.10]   # month-to-month / one year / two year
    elif t <= 36:
        p = [0.45, 0.30, 0.25]
    else:
        p = [0.20, 0.30, 0.50]
    contract_type.append(rng.choice(["Month-to-month", "One year", "Two year"], p=p))
contract_type = np.array(contract_type)

# ---- Other service features --------------------------------------------------
payment_method = rng.choice(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    size=N, p=[0.35, 0.20, 0.25, 0.20],
)
internet_service = rng.choice(["DSL", "Fiber optic", "No"], size=N, p=[0.35, 0.45, 0.20])
tech_support = np.where(rng.random(N) < 0.45, "Yes", "No")
senior_citizen = (rng.random(N) < 0.18).astype(int)

# ---- Monthly charges: fiber costs more; long contracts get slight discount ---
base = np.where(internet_service == "Fiber optic", 75.0,
       np.where(internet_service == "DSL", 55.0, 30.0))
discount = np.where(contract_type == "Two year", 0.92,
            np.where(contract_type == "One year", 0.96, 1.0))
monthly_charges = np.round(base * discount + rng.normal(0, 6, N), 2)
monthly_charges = np.clip(monthly_charges, 18.0, 120.0)

# ---- Total charges ~= tenure * monthly with noise ----------------------------
total_charges = np.round(tenure_months * monthly_charges * rng.uniform(0.95, 1.05, N), 2)

# ---- Signup month (for trend analysis): spread over the last ~8 years --------
# Older cohorts have higher tenure; derive signup month from tenure.
offset_months = tenure_months + rng.integers(0, 24, N)
ref_y, ref_m = 2026, 10
total = ref_y * 12 + (ref_m - 1) - offset_months
sy, sm = total // 12, total % 12 + 1
signup_month = np.array([f"{y:04d}-{m:02d}" for y, m in zip(sy, sm)])

# ---- Churn probability (logistic-style scoring) -------------------------------
score = (
    -2.2
    + 1.6 * (contract_type == "Month-to-month")
    - 0.5 * (contract_type == "One year")
    - 0.9 * (contract_type == "Two year")
    + 0.8 * (tech_support == "No")
    - 0.045 * tenure_months                       # longer tenure -> loyal
    + 0.018 * (monthly_charges - 65)              # pricier plans -> churn
    + 0.35 * senior_citizen
    + 0.25 * (payment_method == "Electronic check")
)
prob = 1 / (1 + np.exp(-score))
churn = (rng.random(N) < prob).astype(int)

df = pd.DataFrame({
    "customer_id": customer_id,
    "signup_month": signup_month,
    "tenure_months": tenure_months,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "contract_type": contract_type,
    "payment_method": payment_method,
    "internet_service": internet_service,
    "tech_support": tech_support,
    "senior_citizen": senior_citizen,
    "churn": churn,
})

df.to_csv(OUT, index=False)
print(f"Wrote {OUT}: {len(df):,} rows")
print(f"Overall churn rate: {df.churn.mean():.1%}")
print(df.groupby("contract_type").churn.mean().round(3))
