"""
backend/tests/test_evaluation.py

Unit tests for EvaluationService.

Covers:
    - Latency ring buffer recording and p95 computation
    - Insufficient-sample guard for p95
    - R@k and MRR computation
    - Citation accuracy computation
    - Hallucination rate computation
    - Missing-label / empty-set error paths
    - report() aggregation
"""

from __future__ import annotations

import pytest

from backend.evaluation import EvaluationService, MetricResult, MetricsReport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_eval_entry(
    retrieved=None,
    relevant_ids=None,
    citations=None,
    reference_pages=None,
    generated_answer=None,
    verified_claims=None,
):
    """Build a minimal eval-set entry with sensible defaults."""
    return {
        "query": "test query",
        "retrieved_chunks": retrieved if retrieved is not None else [],
        "relevant_chunk_ids": relevant_ids if relevant_ids is not None else [],
        "citations": citations if citations is not None else [],
        "reference_source_pages": reference_pages if reference_pages is not None else [],
        "generated_answer": generated_answer if generated_answer is not None else "",
        "verified_claims": verified_claims if verified_claims is not None else [],
    }


def _make_10_entries(**kwargs):
    """Return 10 identical eval entries (minimum required by Req 10.6)."""
    return [_make_eval_entry(**kwargs) for _ in range(10)]


# ---------------------------------------------------------------------------
# MetricResult dataclass
# ---------------------------------------------------------------------------

class TestMetricResult:
    def test_defaults(self):
        r = MetricResult(value=0.5, sample_size=10)
        assert r.error is None
        assert r.is_insufficient is False

    def test_insufficient_flag(self):
        r = MetricResult(value=None, sample_size=5, is_insufficient=True)
        assert r.value is None
        assert r.is_insufficient is True

    def test_error_field(self):
        r = MetricResult(value=None, sample_size=0, error="something went wrong")
        assert r.error == "something went wrong"


# ---------------------------------------------------------------------------
# Latency recording (Requirement 9.3)
# ---------------------------------------------------------------------------

class TestRecordLatencyMs:
    def test_records_positive_value(self):
        svc = EvaluationService()
        svc.record_latency_ms(250.0)
        assert len(svc._latency_buf) == 1
        assert svc._latency_buf[0] == 250.0

    def test_records_multiple_values(self):
        svc = EvaluationService()
        for ms in [100, 200, 300]:
            svc.record_latency_ms(ms)
        assert list(svc._latency_buf) == [100.0, 200.0, 300.0]

    def test_negative_value_clamped_to_zero(self):
        svc = EvaluationService()
        svc.record_latency_ms(-50)
        assert svc._latency_buf[0] == 0.0

    def test_zero_value(self):
        svc = EvaluationService()
        svc.record_latency_ms(0)
        assert svc._latency_buf[0] == 0.0


# ---------------------------------------------------------------------------
# p95 latency — insufficient sample (Requirement 10.2)
# ---------------------------------------------------------------------------

class TestP95InsufficientSample:
    def test_zero_samples_insufficient(self):
        svc = EvaluationService()
        result = svc.p95_latency_s()
        assert result.value is None
        assert result.is_insufficient is True
        assert result.sample_size == 0

    def test_29_samples_insufficient(self):
        svc = EvaluationService()
        for i in range(29):
            svc.record_latency_ms(float(i * 100))
        result = svc.p95_latency_s()
        assert result.value is None
        assert result.is_insufficient is True
        assert result.sample_size == 29

    def test_exactly_30_samples_sufficient(self):
        svc = EvaluationService()
        for i in range(30):
            svc.record_latency_ms(1000.0)
        result = svc.p95_latency_s()
        assert result.value is not None
        assert result.is_insufficient is False


# ---------------------------------------------------------------------------
# p95 latency — value computation (Requirement 10.1)
# ---------------------------------------------------------------------------

class TestP95LatencyValue:
    def test_uniform_samples_p95_matches(self):
        """With 100 samples of 1000 ms each, p95 = 1.0 s."""
        svc = EvaluationService()
        for _ in range(100):
            svc.record_latency_ms(1000.0)
        result = svc.p95_latency_s()
        assert result.value == pytest.approx(1.0, abs=1e-9)
        assert result.sample_size == 100

    def test_p95_value_in_seconds(self):
        """p95 of [100, 200, …, 3000] ms: 95th percentile of 30 values."""
        import math
        svc = EvaluationService()
        samples_ms = [float((i + 1) * 100) for i in range(30)]  # 100..3000
        for ms in samples_ms:
            svc.record_latency_ms(ms)
        result = svc.p95_latency_s()
        # nearest-rank: ceil(0.95 * 30) = 29 → samples_ms[28] = 2900 ms = 2.9 s
        expected_s = round(2900.0 / 1000.0, 2)
        assert result.value == pytest.approx(expected_s, abs=1e-9)

    def test_p95_returns_at_least_2_decimal_places(self):
        """Returned value should be expressible with ≥2 decimal places."""
        svc = EvaluationService()
        for _ in range(30):
            svc.record_latency_ms(1500.0)
        result = svc.p95_latency_s()
        # 1500 ms = 1.5 s; round to 2 dp → 1.5 which equals round(..., 2)
        assert result.value == pytest.approx(1.5, abs=1e-6)

    def test_sample_size_annotated(self):
        svc = EvaluationService()
        for _ in range(50):
            svc.record_latency_ms(500.0)
        result = svc.p95_latency_s()
        assert result.sample_size == 50


# ---------------------------------------------------------------------------
# R@k and MRR (Requirement 10.3)
# ---------------------------------------------------------------------------

class TestRecallAtKAndMRR:
    def test_perfect_retrieval(self):
        """All relevant chunks retrieved → R@k = 1.0, MRR = 1.0."""
        entries = [
            _make_eval_entry(
                retrieved=["a", "b"],
                relevant_ids=["a", "b"],
            )
            for _ in range(10)
        ]
        svc = EvaluationService()
        result = svc.recall_at_k_and_mrr(entries)
        assert result.value == pytest.approx(1.0)
        assert svc._latest_mrr.value == pytest.approx(1.0)

    def test_no_retrieval(self):
        """Nothing retrieved → R@k = 0.0, MRR = 0.0."""
        entries = [
            _make_eval_entry(retrieved=[], relevant_ids=["a"])
            for _ in range(10)
        ]
        svc = EvaluationService()
        result = svc.recall_at_k_and_mrr(entries)
        assert result.value == pytest.approx(0.0)
        assert svc._latest_mrr.value == pytest.approx(0.0)

    def test_mrr_second_rank(self):
        """Relevant chunk at rank 2 → MRR = 0.5."""
        entries = [
            _make_eval_entry(
                retrieved=["x", "a"],
                relevant_ids=["a"],
            )
            for _ in range(10)
        ]
        svc = EvaluationService()
        svc.recall_at_k_and_mrr(entries)
        assert svc._latest_mrr.value == pytest.approx(0.5)

    def test_result_in_range(self):
        entries = [
            _make_eval_entry(retrieved=["a"], relevant_ids=["a", "b"])
            for _ in range(10)
        ]
        svc = EvaluationService()
        result = svc.recall_at_k_and_mrr(entries)
        assert 0.0 <= result.value <= 1.0
        assert 0.0 <= svc._latest_mrr.value <= 1.0

    def test_sample_size_annotated(self):
        entries = _make_10_entries(retrieved=["a"], relevant_ids=["a"])
        svc = EvaluationService()
        result = svc.recall_at_k_and_mrr(entries)
        assert result.sample_size == 10

    def test_under_labeled_set_error(self):
        """Only 9 entries → error, not a metric (Requirement 10.6)."""
        entries = [_make_eval_entry(retrieved=["a"], relevant_ids=["a"]) for _ in range(9)]
        svc = EvaluationService()
        result = svc.recall_at_k_and_mrr(entries)
        assert result.value is None
        assert result.error is not None

    def test_empty_set_error(self):
        svc = EvaluationService()
        result = svc.recall_at_k_and_mrr([])
        assert result.value is None
        assert result.error is not None

    def test_missing_relevant_ids_key_error(self):
        """Entries without 'relevant_chunk_ids' produce an error."""
        entries = [{"query": "q", "retrieved_chunks": ["a"]} for _ in range(10)]
        svc = EvaluationService()
        result = svc.recall_at_k_and_mrr(entries)
        assert result.value is None
        assert result.error is not None


# ---------------------------------------------------------------------------
# Citation accuracy (Requirement 10.4)
# ---------------------------------------------------------------------------

class TestCitationAccuracy:
    def _citation(self, pages):
        """Return a simple dict simulating a Citation object."""
        return {"pages": pages}

    def test_all_citations_on_labeled_pages(self):
        entries = [
            _make_eval_entry(
                citations=[self._citation([1, 2])],
                reference_pages=[1, 2, 3],
            )
            for _ in range(10)
        ]
        svc = EvaluationService()
        result = svc.citation_accuracy(entries)
        assert result.value == pytest.approx(1.0)

    def test_no_citations_on_labeled_pages(self):
        entries = [
            _make_eval_entry(
                citations=[self._citation([5])],
                reference_pages=[1, 2, 3],
            )
            for _ in range(10)
        ]
        svc = EvaluationService()
        result = svc.citation_accuracy(entries)
        assert result.value == pytest.approx(0.0)

    def test_partial_accuracy(self):
        """2 out of 4 cited pages are labeled → accuracy = 0.5."""
        entries = [
            _make_eval_entry(
                citations=[self._citation([1, 2, 5, 6])],
                reference_pages=[1, 2],
            )
            for _ in range(10)
        ]
        svc = EvaluationService()
        result = svc.citation_accuracy(entries)
        assert result.value == pytest.approx(0.5)

    def test_no_citations_returns_one(self):
        """No citations generated → accuracy is 1.0 (nothing wrong)."""
        entries = _make_10_entries(citations=[], reference_pages=[1])
        svc = EvaluationService()
        result = svc.citation_accuracy(entries)
        assert result.value == pytest.approx(1.0)

    def test_result_in_range(self):
        entries = [
            _make_eval_entry(
                citations=[self._citation([1])],
                reference_pages=[1],
            )
            for _ in range(10)
        ]
        svc = EvaluationService()
        result = svc.citation_accuracy(entries)
        assert 0.0 <= result.value <= 1.0

    def test_under_labeled_set_error(self):
        entries = [_make_eval_entry() for _ in range(9)]
        svc = EvaluationService()
        result = svc.citation_accuracy(entries)
        assert result.value is None
        assert result.error is not None

    def test_empty_set_error(self):
        svc = EvaluationService()
        result = svc.citation_accuracy([])
        assert result.value is None
        assert result.error is not None


# ---------------------------------------------------------------------------
# Hallucination rate (Requirement 10.5)
# ---------------------------------------------------------------------------

class TestHallucinationRate:
    def test_zero_hallucination(self):
        """All answer sentences are present in verified_claims → rate = 0."""
        entries = [
            _make_eval_entry(
                generated_answer="The sky is blue.",
                verified_claims=["The sky is blue."],
            )
            for _ in range(10)
        ]
        svc = EvaluationService()
        result = svc.hallucination_rate(entries)
        assert result.value == pytest.approx(0.0)

    def test_full_hallucination(self):
        """No answer sentence matches verified_claims → rate = 1."""
        entries = [
            _make_eval_entry(
                generated_answer="Unicorns exist.",
                verified_claims=["The sky is blue."],
            )
            for _ in range(10)
        ]
        svc = EvaluationService()
        result = svc.hallucination_rate(entries)
        assert result.value == pytest.approx(1.0)

    def test_partial_hallucination(self):
        """Half the answers hallucinate → rate = 0.5."""
        entries = []
        for i in range(10):
            if i < 5:
                entries.append(_make_eval_entry(
                    generated_answer="The sky is blue.",
                    verified_claims=["The sky is blue."],
                ))
            else:
                entries.append(_make_eval_entry(
                    generated_answer="Dragons fly.",
                    verified_claims=["The sky is blue."],
                ))
        svc = EvaluationService()
        result = svc.hallucination_rate(entries)
        assert result.value == pytest.approx(0.5)

    def test_result_in_range(self):
        entries = _make_10_entries(
            generated_answer="Hello world.",
            verified_claims=["Hello world."],
        )
        svc = EvaluationService()
        result = svc.hallucination_rate(entries)
        assert 0.0 <= result.value <= 1.0

    def test_empty_answer_not_hallucinated(self):
        """An empty answer has no claims to check → not hallucinated."""
        entries = _make_10_entries(generated_answer="", verified_claims=[])
        svc = EvaluationService()
        result = svc.hallucination_rate(entries)
        assert result.value == pytest.approx(0.0)

    def test_under_labeled_set_error(self):
        entries = [_make_eval_entry() for _ in range(9)]
        svc = EvaluationService()
        result = svc.hallucination_rate(entries)
        assert result.value is None
        assert result.error is not None

    def test_empty_set_error(self):
        svc = EvaluationService()
        result = svc.hallucination_rate([])
        assert result.value is None
        assert result.error is not None


# ---------------------------------------------------------------------------
# report() (Requirement 10.7)
# ---------------------------------------------------------------------------

class TestReport:
    def test_fresh_instance_report(self):
        """Before any metrics are computed, only p95 is attempted."""
        svc = EvaluationService()
        report = svc.report()
        assert isinstance(report, MetricsReport)
        # p95 is always attempted (may be insufficient)
        assert isinstance(report.p95_latency, MetricResult)
        assert report.p95_latency.is_insufficient is True
        # Retrieval/answer metrics are None until first computation
        assert report.recall_at_k is None
        assert report.mrr is None
        assert report.citation_accuracy is None
        assert report.hallucination_rate is None

    def test_report_reflects_latest_latency(self):
        svc = EvaluationService()
        for _ in range(30):
            svc.record_latency_ms(2000.0)
        report = svc.report()
        assert report.p95_latency.value is not None
        assert report.p95_latency.sample_size == 30

    def test_report_reflects_latest_retrieval_metrics(self):
        svc = EvaluationService()
        entries = _make_10_entries(retrieved=["a"], relevant_ids=["a"])
        svc.recall_at_k_and_mrr(entries)
        report = svc.report()
        assert report.recall_at_k is not None
        assert report.mrr is not None
        assert report.recall_at_k.value == pytest.approx(1.0)

    def test_report_reflects_latest_citation_accuracy(self):
        svc = EvaluationService()
        entries = _make_10_entries(
            citations=[{"pages": [1]}],
            reference_pages=[1],
        )
        svc.citation_accuracy(entries)
        report = svc.report()
        assert report.citation_accuracy is not None
        assert report.citation_accuracy.value == pytest.approx(1.0)

    def test_report_reflects_latest_hallucination_rate(self):
        svc = EvaluationService()
        entries = _make_10_entries(
            generated_answer="Hello.",
            verified_claims=["Hello."],
        )
        svc.hallucination_rate(entries)
        report = svc.report()
        assert report.hallucination_rate is not None
        assert report.hallucination_rate.value == pytest.approx(0.0)

    def test_report_annotates_sample_sizes(self):
        svc = EvaluationService()
        for _ in range(40):
            svc.record_latency_ms(1000.0)
        entries = _make_10_entries(retrieved=["a"], relevant_ids=["a"])
        svc.recall_at_k_and_mrr(entries)
        report = svc.report()
        assert report.p95_latency.sample_size == 40
        assert report.recall_at_k.sample_size == 10


# ---------------------------------------------------------------------------
# Citation dataclass compatibility (Requirement 10.4)
# ---------------------------------------------------------------------------

class TestCitationObjectSupport:
    """citation_accuracy should work with actual Citation dataclass objects."""

    def test_citation_dataclass_with_pages_attribute(self):
        from backend.data_models import Citation
        entries = [
            _make_eval_entry(
                citations=[Citation(filename="doc.pdf", pages=[1, 2])],
                reference_pages=[1, 2],
            )
            for _ in range(10)
        ]
        svc = EvaluationService()
        result = svc.citation_accuracy(entries)
        assert result.value == pytest.approx(1.0)
