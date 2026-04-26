import pandas as pd
import pprint
from fairness_auditor import FairnessAuditor

def test():
    print("Loading data...")
    df = pd.read_csv('../data/supplier_database.csv')
    auditor = FairnessAuditor()

    print("\n=== Task 3: Fairness Metrics ===")
    metrics = auditor.calculate_metrics(df)
    pprint.pprint(metrics)

    print("\n=== Task 4a: Feature Importance Analysis ===")
    importance = auditor.feature_importance_analysis(df)
    pprint.pprint(importance)

    print("\n=== Task 4b: Systemic Bias Validation (Permutation Test) ===")
    systemic = auditor.validate_systemic_bias(df)
    pprint.pprint(systemic)

    print("\n=== Task 4c: Full JSON Explainer Object ===")
    explainer = auditor.generate_explainer(df)
    pprint.pprint(explainer)

    print("\n=== Task 5a: Generate Corrected Scores (Reweighing) ===")
    df_corrected = auditor.generate_corrected_scores(df)
    sample = df_corrected[['supplier_id', 'location_tier', 'ai_trust_score', 'corrected_trust_score']].head(10)
    print(sample.to_string(index=False))

    print("\n=== Task 5b: Before vs After Comparison ===")
    comparison = auditor.compare_before_after(df)
    pprint.pprint(comparison)

if __name__ == "__main__":
    test()
