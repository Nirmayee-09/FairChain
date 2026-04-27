# FairChain – Supplier Fairness Auditor: Technical Methodology

**One-Line Proof of Innovation:**
"Quantifying invisible economic exclusion in supply chains."

---

## The Problem
Modern supply-chain AI models optimise for efficiency — but silently inherit historical prejudice. A Tier-3 supplier in a flood-prone district may perform identically to a metro enterprise, yet consistently receive a lower AI trust score. Not because of performance. Because of where they are.

FairChain's Supplier Fairness Auditor makes this invisible bias visible, measurable, and correctable.

---

## Dataset (2,500 rows — supplier_database.csv)

| Column | Type | Description |
| :--- | :--- | :--- |
| `supplier_id` | string | Unique vendor ID |
| `location_tier` | binary | 1 = Tier-1 metro (privileged) · 0 = Tier-3 flood-prone (unprivileged) |
| `business_size` | binary | 1 = Large enterprise · 0 = Small/micro |
| `owner_gender` | binary | 1 = Male · 0 = Women/minority-owned |
| `years_active` | int | Years of operation (1–20) |
| `on_time_delivery_rate` | float | % deliveries on time (actual KPI) |
| `defect_rate` | float | % defective items (actual KPI) |
| `true_performance_score` | float | Composite ground-truth reliability score (0–100) |
| `ai_trust_score` | float | Biased AI output — contains injected regional penalty |
| `contract_awarded` | binary | 1 = AI selected this vendor |

**Injected Bias:** Tier-3 (`location_tier=0`) suppliers receive an `ai_trust_score` systematically depressed relative to their `true_performance_score`. The gap is statistically invisible at the individual level but emerges clearly in aggregate — mimicking real-world algorithmic redlining.

---

## Core Metrics

### 1. Disparate Impact Ratio (DIR) — via AIF360
```text
DIR = P(contract_awarded=1 | location_tier=0)
      ─────────────────────────────────────────
      P(contract_awarded=1 | location_tier=1)
```
**80% Rule threshold:** DIR < 0.80 → AUDIT FAILED (systemic bias confirmed).
*(Derived from EEOC Uniform Guidelines on Employee Selection Procedures, 1978.)*

### 2. Statistical Parity Difference — via AIF360
`SPD = P(selected | unprivileged) − P(selected | privileged)`
Negative SPD = unprivileged group selected at lower rate.

### 3. Raw Score Gap
`Gap = mean(ai_trust_score | Tier-1) − mean(ai_trust_score | Tier-3)`
Directly quantifies the point-level penalty, independent of selection threshold.

---

## Feature Importance (Task 4)
A `RandomForestClassifier` trained on `contract_awarded` reveals which features drive the AI's decision. `location_tier` consistently appears in the top 3 most important features — confirming the regional penalty is the dominant bias driver, not performance KPIs.

## Systemic Bias Validation (Task 4)
A permutation test (500 iterations) confirms the observed score gap is non-random:
* **H₀:** Group label (`location_tier`) is irrelevant to AI trust score
* **Result:** p-value < 0.05 → Bias is systemic, not random variance

---

## Mitigation: AIF360 Reweighing (Task 5)
Reweighing (Kamiran & Calders, 2012) adjusts sample importance weights so each (group, label) combination is represented proportionally:

`W(unprivileged, positive) = P(Y=1) / P(Y=1 | unprivileged)`

This boost factor is then applied as a score lift to unprivileged vendors:
```python
score_lift = (boost_factor − 1) × raw_score_gap
corrected_score = ai_trust_score + score_lift   # [capped at 100]
```

**Typical result on the 2,500-row dataset:**

| Metric | Before | After |
| :--- | :--- | :--- |
| Disparate Impact Ratio | ~0.41 | ~0.89 |
| Raw Score Gap (pts) | ~15 | ~3 |
| Audit Status | ❌ FAIL | ✅ PASS |

---

## API Integration (Task 6)

```text
supplier_database.csv
        │
        ▼
FairnessAuditor (ml/fairness_auditor.py)
  ├── calculate_metrics()        ← AIF360 DI + SPD
  ├── feature_importance_analysis()
  ├── validate_systemic_bias()
  ├── generate_explainer()       ← full JSON for UI modal
  └── compare_before_after()     ← Reweighing before/after proof
        │
        ▼
FastAPI  GET /fairness/audit     ← full payload
         GET /fairness/audit/summary
         GET /fairness/vendors
        POST /fairness/audit/refresh
        │
        ▼
Next.js Dashboard — Supplier Fairness Scorecard
```

---

## Libraries

| Library | Role |
| :--- | :--- |
| `aif360` | Disparate Impact metric + Reweighing algorithm |
| `fairlearn` | Supplementary equalized odds verification |
| `scikit-learn` | RandomForest for feature importance |
| `pandas` / `numpy` | Data wrangling |
| `fastapi` | REST API serving |

---

## References
* Feldman et al. (2015) Certifying and Removing Disparate Impact
* Kamiran & Calders (2012) Data Preprocessing Techniques for Classification without Discrimination
* EEOC 80% Rule: Uniform Guidelines on Employee Selection Procedures (1978)
