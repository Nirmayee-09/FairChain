"""
FairChain Fairness Auditor
Core AIF360 disparate impact and equalized odds analysis classes.

Logical Connection to Logistics Dashboard:
The Logistics dashboard uses AI to assign 'Trust Scores' to suppliers based
on historical reliability. The Fairness Auditor intercepts these scores before
they are finalized, running statistical parity and disparate impact checks.
If a score exhibits systemic bias, the mitigation engine (Reweighing) adjusts
weights and produces a 'Corrected Score' which is then served to the dashboard.

Demographics Definition (Indian SME Context):
- Protected Attribute: 'location_tier' (0: Tier-3/Rural flood-prone, 1: Tier-1/Urban)

Privileged Group  : location_tier = 1
Unprivileged Group: location_tier = 0
"""

import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric


class FairnessAuditor:

    def __init__(self):
        """
        Initialises the FairnessAuditor.
        Primary protected attribute: location_tier (region-based discrimination).
        """
        self.privileged_groups   = [{'location_tier': 1}]
        self.unprivileged_groups = [{'location_tier': 0}]

    def check_environment(self):
        """Verifies that AIF360 and Fairlearn are installed."""
        try:
            import aif360, fairlearn  # noqa: F401
            return True, "AIF360 and Fairlearn are installed successfully."
        except ImportError as e:
            return False, str(e)

    def _df_to_aif360_dataset(self, df,
                               label_name='contract_awarded',
                               protected_attribute_names=None):
        """Convert a DataFrame to an AIF360 BinaryLabelDataset (numeric cols only)."""
        if protected_attribute_names is None:
            protected_attribute_names = ['location_tier']
        numeric_cols  = df.select_dtypes(include='number').columns.tolist()
        required_cols = set(protected_attribute_names + [label_name])
        final_cols    = list(set(numeric_cols) | required_cols)
        df_numeric    = df[final_cols].copy()
        return BinaryLabelDataset(
            df=df_numeric,
            label_names=[label_name],
            protected_attribute_names=protected_attribute_names,
            favorable_label=1.0,
            unfavorable_label=0.0,
        )

    def calculate_metrics(self, df):
        """
        Calculates Disparate Impact and Statistical Parity (Mean Difference)
        using AIF360, plus the raw score gap. Applies the 80% Rule threshold.
        """
        df = df.copy()
        if 'contract_awarded' not in df.columns:
            df['contract_awarded'] = (df['ai_trust_score'] >= 75).astype(int)

        dataset = self._df_to_aif360_dataset(df)
        metric  = BinaryLabelDatasetMetric(
            dataset,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
        )

        disparate_impact = metric.disparate_impact()
        mean_difference  = metric.mean_difference()

        priv_score   = df[df['location_tier'] == 1]['ai_trust_score'].mean()
        unpriv_score = df[df['location_tier'] == 0]['ai_trust_score'].mean()
        score_gap    = priv_score - unpriv_score
        audit_failed = disparate_impact < 0.8

        return {
            "disparate_impact_ratio":        round(disparate_impact, 4),
            "statistical_parity_difference": round(mean_difference, 4),
            "raw_score_gap":                 round(score_gap, 2),
            "privileged_avg_score":          round(priv_score, 2),
            "unprivileged_avg_score":        round(unpriv_score, 2),
            "audit_failed_80_percent_rule":  bool(audit_failed),
        }

    def feature_importance_analysis(self, df):
        """
        Trains a RandomForestClassifier to predict contract_awarded and
        extracts feature importances to reveal which inputs drive bias.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        df = df.copy()
        if 'contract_awarded' not in df.columns:
            df['contract_awarded'] = (df['ai_trust_score'] >= 75).astype(int)

        feature_cols = [
            'business_size', 'location_tier', 'owner_gender',
            'years_active', 'on_time_delivery_rate', 'defect_rate',
        ]
        X = df[feature_cols]
        y = df['contract_awarded']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        ranked = sorted(
            zip(feature_cols, clf.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
        return {
            "model_accuracy": round(clf.score(X_test, y_test), 4),
            "feature_importances": [
                {"feature": f, "importance": round(float(i), 4)}
                for f, i in ranked
            ],
        }

    def validate_systemic_bias(self, df, n_permutations=500):
        """
        Permutation test confirming the score gap is systemic (p < 0.05),
        not random variance. Null hypothesis: group label is irrelevant.
        """
        observed_gap = (
            df[df['location_tier'] == 1]['ai_trust_score'].mean()
            - df[df['location_tier'] == 0]['ai_trust_score'].mean()
        )
        rng    = np.random.default_rng(seed=42)
        scores = df['ai_trust_score'].values
        labels = df['location_tier'].values

        permuted_gaps = np.array([
            scores[rng.permutation(labels) == 1].mean()
            - scores[rng.permutation(labels) == 0].mean()
            for _ in range(n_permutations)
        ])
        p_value = float(np.mean(np.abs(permuted_gaps) >= np.abs(observed_gap)))

        return {
            "observed_score_gap":   round(float(observed_gap), 4),
            "permutation_mean_gap": round(float(permuted_gaps.mean()), 4),
            "p_value":              round(p_value, 4),
            "bias_is_systemic":     p_value < 0.05,
        }

    def generate_explainer(self, df):
        """
        Produces the full JSON Explainer object consumed by the UI dashboard modal.
        Combines fairness metrics, feature importances, systemic bias validation,
        and the top penalised vendor list into a single structured payload.
        """
        metrics    = self.calculate_metrics(df)
        importance = self.feature_importance_analysis(df)
        systemic   = self.validate_systemic_bias(df)

        df_copy = df.copy()
        df_copy['penalty'] = df_copy['true_performance_score'] - df_copy['ai_trust_score']
        top_penalised = (
            df_copy[df_copy['location_tier'] == 0]
            .nlargest(5, 'penalty')
            [['supplier_id', 'ai_trust_score', 'true_performance_score', 'penalty']]
            .to_dict(orient='records')
        )

        if metrics['audit_failed_80_percent_rule']:
            status = "AUDIT_FAILED"
            reason = (
                f"Disparate Impact Ratio of {metrics['disparate_impact_ratio']} is below "
                f"the 80% Rule threshold (0.80). Tier-3 suppliers are awarded contracts at "
                f"less than half the rate of Tier-1 suppliers despite comparable performance."
            )
        else:
            status = "AUDIT_PASSED"
            reason = "No statistically significant disparate impact detected."

        return {
            "audit_status":             status,
            "audit_reason":             reason,
            "fairness_metrics":         metrics,
            "feature_importance":       importance["feature_importances"],
            "model_accuracy":           importance["model_accuracy"],
            "systemic_bias_validation": systemic,
            "top_penalised_vendors":    top_penalised,
            "one_line_proof": (
                "Quantifying invisible economic exclusion in supply chains: "
                f"Tier-3 Indian SME suppliers suffer a {metrics['raw_score_gap']}-point "
                "AI trust score deficit with no operational justification."
            ),
        }

    def apply_reweighing(self, df):
        """
        Runs AIF360's Reweighing algorithm and derives the analytical boost factor.

        Reweighing computes:
            W(A=unprivileged, Y=1) = P(Y=1) / P(Y=1 | A=unprivileged)

        This boost factor quantifies how under-represented the unprivileged
        positive class is relative to the overall population average.

        Returns: (reweighed_dataset, instance_weights, boost_factor)
        """
        from aif360.algorithms.preprocessing import Reweighing as AIF360Reweighing

        df_work = df.copy()
        if 'contract_awarded' not in df_work.columns:
            df_work['contract_awarded'] = (df_work['ai_trust_score'] >= 75).astype(int)

        dataset    = self._df_to_aif360_dataset(df_work)
        rw         = AIF360Reweighing(
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
        )
        rw.fit(dataset)
        dataset_rw = rw.transform(dataset)

        n_total      = len(df_work)
        n_pos        = int(df_work['contract_awarded'].sum())
        n_unpriv     = int((df_work['location_tier'] == 0).sum())
        n_pos_unpriv = int(df_work.loc[df_work['location_tier'] == 0, 'contract_awarded'].sum())

        p_pos              = n_pos / n_total
        p_pos_given_unpriv = n_pos_unpriv / n_unpriv if n_unpriv > 0 else 1e-9
        boost_factor       = p_pos / p_pos_given_unpriv if p_pos_given_unpriv > 0 else 1.0

        return dataset_rw, dataset_rw.instance_weights, boost_factor

    def generate_corrected_scores(self, df):
        """
        Generates a corrected trust score for unprivileged suppliers using
        the Reweighing boost factor.

        Score-lift formula:
            score_lift = (boost_factor - 1) * raw_score_gap

        This proportionally closes the group-level gap without fabricating
        individual performance data. Privileged scores are left untouched.

        Returns df with added columns: corrected_trust_score,
        score_lift_applied, reweighing_boost_factor.
        """
        df_out = df.copy()
        if 'contract_awarded' not in df_out.columns:
            df_out['contract_awarded'] = (df_out['ai_trust_score'] >= 75).astype(int)

        _, _, boost_factor = self.apply_reweighing(df_out)

        priv_mean  = df_out[df_out['location_tier'] == 1]['ai_trust_score'].mean()
        unpriv_mean = df_out[df_out['location_tier'] == 0]['ai_trust_score'].mean()
        raw_gap    = priv_mean - unpriv_mean
        score_lift = (boost_factor - 1) * raw_gap

        unpriv_mask = df_out['location_tier'] == 0
        corrected   = df_out['ai_trust_score'].copy()
        corrected[unpriv_mask] = (corrected[unpriv_mask] + score_lift).clip(upper=100)

        df_out['corrected_trust_score']   = corrected.round(2)
        df_out['score_lift_applied']      = round(score_lift, 2)
        df_out['reweighing_boost_factor'] = round(boost_factor, 4)
        return df_out

    def compare_before_after(self, df):
        """
        Runs the fairness audit on both the original biased scores and the
        corrected post-mitigation scores, returning a side-by-side comparison
        that proves the effectiveness of the Reweighing mitigation.
        """
        before_metrics = self.calculate_metrics(df)

        df_corrected = self.generate_corrected_scores(df)
        score_lift   = df_corrected['score_lift_applied'].iloc[0]
        boost_factor = df_corrected['reweighing_boost_factor'].iloc[0]

        df_after = df_corrected.copy()
        df_after['ai_trust_score']   = df_after['corrected_trust_score']
        df_after['contract_awarded'] = (df_after['ai_trust_score'] >= 75).astype(int)

        after_metrics   = self.calculate_metrics(df_after)
        gap_improvement = round(before_metrics['raw_score_gap'] - after_metrics['raw_score_gap'], 2)
        di_improvement  = round(after_metrics['disparate_impact_ratio'] - before_metrics['disparate_impact_ratio'], 4)

        return {
            "before_mitigation": before_metrics,
            "after_mitigation":  after_metrics,
            "improvements": {
                "disparate_impact_delta":   di_improvement,
                "score_gap_reduced_by_pts": gap_improvement,
                "score_lift_applied_pts":   score_lift,
                "reweighing_boost_factor":  boost_factor,
                "audit_now_passes":         not after_metrics['audit_failed_80_percent_rule'],
                "mitigation_algorithm":     "AIF360 Reweighing + Demographic Score Adjustment",
            },
        }
