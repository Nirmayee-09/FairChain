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
    

import time
import numpy as np
import pytest


def _make_df(n=600, bias=True, seed=42):
    """
    Synthetic DataFrame matching supplier_database.csv schema.
    bias=True  → Tier-3 suppliers get ai_trust_score ~15 pts below true_performance
    bias=False → scores track performance closely (fair data, DI should pass)
    """
    rng = np.random.default_rng(seed)
    n_priv, n_unpriv = n // 2, n - n // 2

    def _group(tier, size, penalty):
        true_score = rng.uniform(40, 90, size)
        ai_score   = np.clip(true_score - penalty + rng.normal(0, 2, size), 0, 100)
        return pd.DataFrame({
            "supplier_id":            [f"SUP-{tier}-{i:04d}" for i in range(size)],
            "business_size":          rng.integers(0, 2, size),
            "location_tier":          tier,
            "owner_gender":           rng.integers(0, 2, size),
            "years_active":           rng.integers(1, 21, size),
            "on_time_delivery_rate":  rng.uniform(70, 99, size).round(2),
            "defect_rate":            rng.uniform(0.5, 5.0, size).round(2),
            "true_performance_score": true_score.round(2),
            "ai_trust_score":         ai_score.round(2),
            "contract_awarded":       (ai_score >= 75).astype(int),
        })

    return pd.concat(
        [_group(1, n_priv, 0.0), _group(0, n_unpriv, 15.0 if bias else 0.5)],
        ignore_index=True,
    )


# ── Guardrails ────────────────────────────────────────────────────

class TestGuardrails:

    def test_missing_location_tier_raises(self):
        """calculate_metrics must fail cleanly if location_tier is absent."""
        df = _make_df().drop(columns=["location_tier"])
        with pytest.raises(Exception):
            FairnessAuditor().calculate_metrics(df)

    def test_missing_ai_trust_score_raises(self):
        """calculate_metrics must fail cleanly if ai_trust_score is absent."""
        df = _make_df().drop(columns=["ai_trust_score"])
        with pytest.raises(Exception):
            FairnessAuditor().calculate_metrics(df)

    def test_missing_feature_col_raises(self):
        """feature_importance_analysis must fail if a feature column is absent."""
        df = _make_df().drop(columns=["on_time_delivery_rate"])
        with pytest.raises(Exception):
            FairnessAuditor().feature_importance_analysis(df)

    def test_missing_true_performance_raises(self):
        """generate_explainer uses true_performance_score — must fail if missing."""
        df = _make_df().drop(columns=["true_performance_score"])
        with pytest.raises(Exception):
            FairnessAuditor().generate_explainer(df)

    def test_empty_dataframe_raises(self):
        """All public methods must fail gracefully on empty DataFrames."""
        df = _make_df().iloc[0:0]
        with pytest.raises(Exception):
            FairnessAuditor().calculate_metrics(df)


# ── Metric Correctness ────────────────────────────────────────────

class TestMetricCorrectness:

    def test_biased_data_fails_80_percent_rule(self):
        metrics = FairnessAuditor().calculate_metrics(_make_df(bias=True))
        assert metrics["audit_failed_80_percent_rule"] is True
        assert metrics["disparate_impact_ratio"] < 0.80

    def test_fair_data_passes_80_percent_rule(self):
        metrics = FairnessAuditor().calculate_metrics(_make_df(bias=False, n=1000))
        assert metrics["audit_failed_80_percent_rule"] is False
        assert metrics["disparate_impact_ratio"] >= 0.80

    def test_parity_gap_positive_for_biased_data(self):
        metrics = FairnessAuditor().calculate_metrics(_make_df(bias=True))
        assert metrics["raw_score_gap"] > 0

    def test_all_metric_keys_present(self):
        metrics = FairnessAuditor().calculate_metrics(_make_df())
        for key in [
            "disparate_impact_ratio", "statistical_parity_difference",
            "raw_score_gap", "privileged_avg_score",
            "unprivileged_avg_score", "audit_failed_80_percent_rule",
        ]:
            assert key in metrics, f"Missing key: {key}"

    def test_systemic_bias_detected_on_biased_data(self):
        result = FairnessAuditor().validate_systemic_bias(_make_df(bias=True, n=800), n_permutations=300)
        assert result["bias_is_systemic"] is True
        assert result["p_value"] < 0.05

    def test_no_systemic_bias_on_fair_data(self):
        result = FairnessAuditor().validate_systemic_bias(_make_df(bias=False, n=800), n_permutations=300)
        assert result["bias_is_systemic"] is False


# ── Mitigation Correctness ───────────────────────────────────────

class TestMitigationCorrectness:

    def test_boost_factor_gt_1_for_biased_data(self):
        _, _, boost = FairnessAuditor().apply_reweighing(_make_df(bias=True))
        assert boost > 1.0

    def test_corrected_scores_gte_original_for_unprivileged(self):
        df_corr = FairnessAuditor().generate_corrected_scores(_make_df(bias=True))
        tier3 = df_corr[df_corr["location_tier"] == 0]
        assert (tier3["corrected_trust_score"] >= tier3["ai_trust_score"]).all()

    def test_corrected_scores_capped_at_100(self):
        df_corr = FairnessAuditor().generate_corrected_scores(_make_df(bias=True))
        assert (df_corr["corrected_trust_score"] <= 100).all()

    def test_privileged_scores_unchanged(self):
        df_corr = FairnessAuditor().generate_corrected_scores(_make_df(bias=True))
        tier1 = df_corr[df_corr["location_tier"] == 1]
        pd.testing.assert_series_equal(
            tier1["ai_trust_score"].reset_index(drop=True),
            tier1["corrected_trust_score"].reset_index(drop=True),
            check_names=False,
        )

    def test_di_improves_after_mitigation(self):
        result = FairnessAuditor().compare_before_after(_make_df(bias=True))
        assert result["after_mitigation"]["disparate_impact_ratio"] > \
               result["before_mitigation"]["disparate_impact_ratio"]
        assert result["improvements"]["disparate_impact_delta"] > 0

    def test_audit_passes_after_mitigation(self):
        result = FairnessAuditor().compare_before_after(_make_df(bias=True, n=1000))
        assert result["improvements"]["audit_now_passes"] is True


# ── Explainer JSON contract (for Jaideep's Dashboard Modal) ───────

class TestExplainerContract:

    def test_all_top_level_keys_present(self):
        explainer = FairnessAuditor().generate_explainer(_make_df(bias=True))
        for key in [
            "audit_status", "audit_reason", "fairness_metrics",
            "feature_importance", "model_accuracy",
            "systemic_bias_validation", "top_penalised_vendors", "one_line_proof",
        ]:
            assert key in explainer, f"Explainer missing key: {key}"

    def test_biased_status_is_audit_failed(self):
        explainer = FairnessAuditor().generate_explainer(_make_df(bias=True))
        assert explainer["audit_status"] == "AUDIT_FAILED"

    def test_fair_status_is_audit_passed(self):
        explainer = FairnessAuditor().generate_explainer(_make_df(bias=False, n=1000))
        assert explainer["audit_status"] == "AUDIT_PASSED"

    def test_top_penalised_vendors_non_empty(self):
        explainer = FairnessAuditor().generate_explainer(_make_df(bias=True))
        assert len(explainer["top_penalised_vendors"]) > 0

    def test_vendor_records_have_required_fields(self):
        explainer = FairnessAuditor().generate_explainer(_make_df(bias=True))
        for v in explainer["top_penalised_vendors"]:
            for field in ["supplier_id", "ai_trust_score", "true_performance_score", "penalty"]:
                assert field in v, f"Vendor record missing field: {field}"

    def test_feature_importance_is_ranked_list(self):
        explainer = FairnessAuditor().generate_explainer(_make_df(bias=True))
        fi = explainer["feature_importance"]
        assert isinstance(fi, list) and len(fi) > 0
        assert "feature" in fi[0] and "importance" in fi[0]


# ── Performance — up to 5,000 rows under 10 seconds ──────────────

class TestPerformance:

    @pytest.mark.parametrize("n_rows", [500, 2000, 5000])
    def test_full_pipeline_within_time_limit(self, n_rows):
        df = _make_df(n=n_rows, bias=True)
        auditor = FairnessAuditor()
        start = time.perf_counter()
        auditor.calculate_metrics(df)
        auditor.generate_explainer(df)
        auditor.compare_before_after(df)
        elapsed = time.perf_counter() - start
        assert elapsed < 10.0, f"Full audit for {n_rows} rows took {elapsed:.2f}s — exceeds 10s"


# ── Environment check (libraries importable) ─────────────────────

class TestEnvironment:

    def test_aif360_and_fairlearn_installed(self):
        ok, msg = FairnessAuditor().check_environment()
        assert ok, f"Library check failed: {msg}"
