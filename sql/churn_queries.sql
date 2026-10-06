-- =====================================================================
-- Telecom Customer Churn Analysis — Advanced SQL
-- Dataset: data/customers.csv (import into a table named `customers`)
-- Covers: CTEs, window functions, conditional aggregation, ranking
-- =====================================================================

-- ---------------------------------------------------------------------
-- Q1. Churn rate by contract type (with share of customer base)
-- ---------------------------------------------------------------------
WITH contract_stats AS (
    SELECT
        contract_type,
        COUNT(*)                                   AS customers,
        SUM(churn)                                 AS churned
    FROM customers
    GROUP BY contract_type
)
SELECT
    contract_type,
    customers,
    churned,
    ROUND(100.0 * churned / customers, 2)          AS churn_rate_pct,
    ROUND(100.0 * customers / SUM(customers) OVER (), 2) AS share_of_base_pct
FROM contract_stats
ORDER BY churn_rate_pct DESC;

-- ---------------------------------------------------------------------
-- Q2. Average tenure of churned vs retained customers (by contract type)
-- ---------------------------------------------------------------------
SELECT
    contract_type,
    CASE WHEN churn = 1 THEN 'Churned' ELSE 'Retained' END AS status,
    COUNT(*)                        AS customers,
    ROUND(AVG(tenure_months), 1)     AS avg_tenure_months,
    ROUND(AVG(monthly_charges), 2)   AS avg_monthly_charges
FROM customers
GROUP BY contract_type, churn
ORDER BY contract_type, churn;

-- ---------------------------------------------------------------------
-- Q3. Rolling 3-month churn trend by signup cohort
-- Uses a window function to smooth monthly cohort churn rates.
-- ---------------------------------------------------------------------
WITH monthly_churn AS (
    SELECT
        signup_month,
        COUNT(*)                    AS cohort_size,
        SUM(churn)                  AS churned,
        1.0 * SUM(churn) / COUNT(*) AS churn_rate
    FROM customers
    GROUP BY signup_month
)
SELECT
    signup_month,
    cohort_size,
    ROUND(churn_rate * 100, 2) AS churn_rate_pct,
    ROUND(AVG(churn_rate) OVER (
        ORDER BY signup_month
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) * 100, 2) AS rolling_3m_churn_rate_pct
FROM monthly_churn
ORDER BY signup_month;

-- ---------------------------------------------------------------------
-- Q4. Top risk segments: churn rate by contract x tech support x senior
-- Ranked with a window function; segments with < 50 customers filtered out
-- for statistical reliability.
-- ---------------------------------------------------------------------
WITH segments AS (
    SELECT
        contract_type,
        tech_support,
        senior_citizen,
        internet_service,
        COUNT(*)                    AS customers,
        SUM(churn)                  AS churned,
        1.0 * SUM(churn) / COUNT(*) AS churn_rate
    FROM customers
    GROUP BY contract_type, tech_support, senior_citizen, internet_service
    HAVING COUNT(*) >= 50
)
SELECT
    contract_type,
    tech_support,
    senior_citizen,
    internet_service,
    customers,
    churned,
    ROUND(churn_rate * 100, 2) AS churn_rate_pct,
    RANK() OVER (ORDER BY churn_rate DESC) AS risk_rank
FROM segments
ORDER BY risk_rank
LIMIT 10;

-- ---------------------------------------------------------------------
-- Q5. Revenue at risk: monthly revenue exposed by churned customers,
--     plus the "saveable" share on month-to-month contracts
-- ---------------------------------------------------------------------
WITH revenue AS (
    SELECT
        contract_type,
        SUM(CASE WHEN churn = 1 THEN monthly_charges ELSE 0 END) AS churned_mrr,
        SUM(monthly_charges)                                     AS total_mrr,
        SUM(CASE WHEN churn = 1 AND contract_type = 'Month-to-month'
                 THEN monthly_charges ELSE 0 END)                 AS mtm_churned_mrr
    FROM customers
    GROUP BY contract_type
)
SELECT
    contract_type,
    ROUND(churned_mrr, 2)                      AS churned_mrr,
    ROUND(total_mrr, 2)                         AS total_mrr,
    ROUND(100.0 * churned_mrr / total_mrr, 2)  AS pct_mrr_lost,
    ROUND(100.0 * SUM(mtm_churned_mrr) OVER ()
                / SUM(churned_mrr) OVER (), 2) AS pct_of_churned_mrr_on_mtm
FROM revenue
ORDER BY churned_mrr DESC;

-- ---------------------------------------------------------------------
-- Q6. Tenure deciles vs churn: where in the customer lifecycle does
--     churn concentrate? (NTILE window function)
-- ---------------------------------------------------------------------
WITH deciles AS (
    SELECT
        churn,
        NTILE(10) OVER (ORDER BY tenure_months) AS tenure_decile
    FROM customers
)
SELECT
    tenure_decile,
    COUNT(*)                    AS customers,
    SUM(churn)                  AS churned,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate_pct
FROM deciles
GROUP BY tenure_decile
ORDER BY tenure_decile;
