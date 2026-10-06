# Telecom Customer Churn Analysis

An end-to-end customer churn analysis for a fictional telecom company — from
SQL exploration to a predictive model — built as a portfolio project.

## Problem Statement

Customer churn is one of the costliest problems in telecom: acquiring a new
subscriber costs far more than retaining an existing one. This project answers
three business questions:

1. **Which customers are churning, and at what rate?**
2. **What drives churn — contract type, support, tenure, or price?**
3. **Which segments should retention campaigns target first?**

## Dataset

`data/customers.csv` — **synthetic data** (8,000 customers, generated with
`generate_data.py`, seed=42 for full reproducibility). No real personal data.

| Column | Description |
|---|---|
| `customer_id` | Unique customer identifier |
| `signup_month` | Cohort month (YYYY-MM, for trend analysis) |
| `tenure_months` | Months as a customer (1–72) |
| `monthly_charges` | Current monthly bill ($) |
| `total_charges` | Lifetime revenue ($) |
| `contract_type` | Month-to-month / One year / Two year |
| `payment_method` | Electronic check, Mailed check, Bank transfer, Credit card |
| `internet_service` | DSL / Fiber optic / No |
| `tech_support` | Yes / No |
| `senior_citizen` | 0 / 1 |
| `churn` | 1 = churned, 0 = retained (target) |

## Methodology

**SQL (`sql/churn_queries.sql`)** — 6 advanced queries using CTEs, window
functions (`AVG() OVER`, `RANK()`, `NTILE`), and conditional aggregation:
churn rate by contract type, churned-vs-retained tenure comparison, rolling
3-month churn trend by signup cohort, top-10 risk segments, revenue at risk,
and churn concentration across tenure deciles.

**Python (`analysis.py`)** — pandas EDA (data-quality checks, groupby churn
drivers, correlations) plus a scikit-learn logistic regression on
one-hot-encoded features (standardized) to rank feature importance.
Holdout evaluation: 80/20 stratified split. Three charts saved to `charts/`.

## Key Findings

1. **Contract type is the dominant churn driver.** Month-to-month customers
   churn at **31.9%**, vs **4.8%** on one-year and **2.4%** on two-year
   contracts — roughly **13x higher** than two-year customers.
2. **Tech support cuts churn nearly in half.** Customers without tech support
   churn at **23.3%** vs **12.9%** with support.
3. **Churn strikes early.** Churned customers average **12.9 months** tenure
   vs **19.7 months** for retained customers; the tenure decile analysis shows
   churn concentrated in the lowest-tenure cohorts.
4. **Price sensitivity is real but secondary.** Churned customers pay
   **$62.47/mo** on average vs **$56.33/mo** for retained customers
   (correlation of monthly charges with churn: +0.135).
5. **Revenue at risk: ~$93K/month.** Churned customers represent
   **$93,147/mo** in lost monthly recurring revenue — the retention business
   case in one number.

The logistic regression (82.1% holdout accuracy) ranks the top predictors as:
**two-year contract, one-year contract, tenure, tech support** — confirming
the EDA story with a model.

## Business Recommendations

1. **Convert month-to-month customers to annual contracts** — offer a
   first-year discount or loyalty perk; this single lever addresses the
   31.9% churn segment.
2. **Bundle free tech support into onboarding** for the first 3 months —
   the cheapest intervention against the 23.3% no-support churn rate.
3. **Launch a 0–12 month "new customer success" program** (check-in calls,
   usage tips) since churn concentrates in early tenure.
4. **Review fiber-optic pricing** — fiber customers churn at 22.5% vs 17.0%
   for DSL, suggesting a value-perception gap at higher price points.

## Tools Used

- **SQL** — CTEs, window functions, conditional aggregation
- **Python** — pandas, scikit-learn (logistic regression), matplotlib
- **Reproducibility** — deterministic data generation (seed=42)

## How to Run

```bash
# 1. (Re)generate the synthetic dataset
~/workspace/.venv/bin/python generate_data.py

# 2. Run the full analysis (prints insights, saves charts)
~/workspace/.venv/bin/python analysis.py

# 3. SQL queries: load data/customers.csv into a `customers` table,
#    then run sql/churn_queries.sql in any SQL engine.
```

## Project Structure

```
telecom-churn-analysis/
├── README.md
├── generate_data.py        # deterministic synthetic data generator
├── analysis.py             # EDA + logistic regression + charts
├── data/
│   └── customers.csv       # 8,000 synthetic customers
├── sql/
│   └── churn_queries.sql   # 6 advanced analytical queries
└── charts/
    ├── churn_by_contract.png
    ├── tenure_distribution.png
    └── feature_importance.png
```
