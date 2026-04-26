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

if __name__ == "__main__":
    test()
