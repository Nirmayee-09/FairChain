"""
FairChain Fairness Auditor
Core AIF360 disparate impact and equalized odds analysis classes.

Logical Connection to Logistics Dashboard:
The Logistics dashboard uses AI to assign 'Trust Scores' to suppliers based on historical reliability.
The Fairness Auditor intercepts these scores before they are finalized, running statistical parity
and disparate impact checks. If a score exhibits systemic bias, the mitigation engine (e.g., Reweighing)
adjusts the weights to produce a 'Corrected Score' which is then sent to the dashboard.

Demographics Definition (Indian SME Context):
- Protected Attribute: 'business_size' (0: Small/Micro Enterprise, 1: Large Enterprise)
- Protected Attribute: 'location_tier' (0: Tier-3/Rural, 1: Tier-1/Urban)
- Protected Attribute: 'owner_gender' (0: Women/Minority, 1: Male)

Privileged Group: 
  - location_tier = 1

Unprivileged Group:
  - location_tier = 0
"""

import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

class FairnessAuditor:
    def __init__(self):
        """
        Initializes the FairnessAuditor with defined privileged and unprivileged groups.
        """
        # For AIF360, we define the groups using the protected attribute dictionary.
        # We'll focus on 'location_tier' as the primary protected attribute for the region penalty,
        # but could easily expand to business_size or owner_gender.
        self.privileged_groups = [{'location_tier': 1}]
        self.unprivileged_groups = [{'location_tier': 0}]
        
    def check_environment(self):
        """
        Verifies that the fairness ML environment is properly set up.
        """
        try:
            import aif360
            import fairlearn
            return True, "AIF360 and Fairlearn are installed successfully."
        except ImportError as e:
            return False, str(e)
            
    def _df_to_aif360_dataset(self, df, label_name='contract_awarded', protected_attribute_names=['location_tier']):
        """
        Converts a Pandas DataFrame to an AIF360 BinaryLabelDataset.
        AIF360 requires ALL columns to be numeric — we select only the needed columns.
        """
        # Select only numeric columns required: features + label + protected attribute
        required_cols = protected_attribute_names + [label_name]
        # Keep all numeric feature cols (drop string IDs)
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        # Make sure required cols are in
        final_cols = list(set(numeric_cols) | set(required_cols))
        df_numeric = df[final_cols].copy()
        
        dataset = BinaryLabelDataset(
            df=df_numeric,
            label_names=[label_name],
            protected_attribute_names=protected_attribute_names,
            favorable_label=1.0,
            unfavorable_label=0.0
        )
        return dataset

    def calculate_metrics(self, df):
        """
        Calculates Disparate Impact and Statistical Parity (Mean Difference) using AIF360.
        """
        # Ensure 'contract_awarded' exists. If not, generate it based on ai_trust_score threshold (e.g. 75)
        if 'contract_awarded' not in df.columns:
            df['contract_awarded'] = (df['ai_trust_score'] >= 75).astype(int)
            
        dataset = self._df_to_aif360_dataset(df)
        
        metric = BinaryLabelDatasetMetric(
            dataset,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups
        )
        
        # 1. Disparate Impact Ratio: (unprivileged success rate / privileged success rate)
        disparate_impact = metric.disparate_impact()
        
        # 2. Statistical Parity / Mean Difference in success rates
        mean_difference = metric.mean_difference()
        
        # Calculate Mean Difference in raw AI Trust Scores directly using pandas
        # This highlights the "unfair gap" in continuous score
        priv_score = df[df['location_tier'] == 1]['ai_trust_score'].mean()
        unpriv_score = df[df['location_tier'] == 0]['ai_trust_score'].mean()
        score_gap = priv_score - unpriv_score
        
        # 3. Apply the '80% Rule' (Four-Fifths Rule)
        # If disparate impact < 0.8, the audit fails
        audit_failed = disparate_impact < 0.8
        
        return {
            "disparate_impact_ratio": round(disparate_impact, 4),
            "statistical_parity_difference": round(mean_difference, 4),
            "raw_score_gap": round(score_gap, 2),
            "privileged_avg_score": round(priv_score, 2),
            "unprivileged_avg_score": round(unpriv_score, 2),
            "audit_failed_80_percent_rule": bool(audit_failed)
        }
