import pandas as pd
from fairness_auditor import FairnessAuditor

def test():
    print("Loading data...")
    df = pd.read_csv('../data/supplier_database.csv')
    
    auditor = FairnessAuditor()
    print("Environment ok?", auditor.check_environment())
    
    print("Calculating metrics...")
    metrics = auditor.calculate_metrics(df)
    
    import pprint
    pprint.pprint(metrics)

if __name__ == "__main__":
    test()
