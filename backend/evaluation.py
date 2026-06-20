"""
backend/evaluation.py

EvaluationService — records per-query latency and computes retrieval/answer
quality metrics for the RAG PDF Chatbot.

Requirements covered:
    9.3  – record end-to-end response time (ms) per handled query
    10.1 – p95 latency when ≥30 samples: seconds with ≥2 decimal places
    10.2 – p95 returns insufficient-sample result when <30 samples
    10.3 – R@k and MRR over labeled eval sets (≥10 queries), in [0, 1]
    10.4 – citation_accuracy = cited-on-labeled-pages / total citations
    10.5 – hallucination_rate = answers-with-unchecked-claims / total answers
    10.6 – empty or under-labeled sets → error, not a metric
    10.7 – report() returns latest values annotated with sample sizes
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    """Holds a single computed metric value together with provenance info.

    Attributes:
        value:           The numeric metric value, or ``None`` when insufficient
                         data prevents computation.
        sample_size:     Number of samples (latency recordings or eval queries)
                         used (or available) for this result.
        error:           Human-readable error string when the metric cannot be
                         computed due to a data-quality problem (missing labels,
                         empty set).  ``None`` when not in error.
        is_insufficient: ``True`` when the sample is below the required minimum
                         (< MIN_LATENCY_SAMPLES for latency, < MIN_EVAL_QUERIES
                         for retrieval/answer metrics).  Distinct from *error*:
                         the data might be valid but simply too sparse.
    """

    value: float | None
    sample_size: int
    error: str | None = None
    is_insufficient: bool = False


@dataclass
class MetricsReport:
    """Snapshot of the latest computed metrics, as returned by ``report()``.

    Fields are ``None`` when the corresponding metric has never been computed
    (e.g. ``recall_at_k`` is ``None`` until the first successful call to
    ``recall_at_k_and_mrr``).
    """

    p95_latency: MetricResult
    recall_at_k: MetricResult | None
    mrr: MetricResult | None
    citation_accuracy: MetricResult | None
    hallucination_rate: MetricResult | None


# ---------------------------------------------------------------------------
# EvaluationService
# ---------------------------------------------------------------------------

class EvaluationService:
    """Records latency samples and computes retrieval/answer quality metrics.

    Thread-safety
    -------------
    ``record_latency_ms`` is protected by a lock so the ring buffer can be
    updated safely from concurrent request handlers.  Metric computations are
    read-only with respect to the ring buffer and are not individually locked
    (callers should not mutate the eval_set while a computation is in flight).

    Latency ring buffer
    -------------------
    Uses ``collections.deque`` with no size limit — all samples are retained.

    Eval set format
    ---------------
    Each entry in *eval_set* is a ``dict`` with the keys listed below.  Keys
    used for each metric are noted inline; absent or ``None`` values trigger the
    missing-label error path.

    .. code-block:: python

        {
            "query": str,
            # R@k / MRR
            "retrieved_chunks": list[str | ChunkMetadata-like],
            "relevant_chunk_ids": list[str],          # labeled relevant chunk ids
            # citation accuracy
            "citations": list[Citation-like],          # Citation objects or dicts
            "reference_source_pages": list[int],       # labeled ground-truth pages
            # hallucination rate
            "generated_answer": str,
            "verified_claims": list[str],              # claims verifiable in chunks
        }
    """

    MIN_LATENCY_SAMPLES: int = 30
    MIN_EVAL_QUERIES: int = 10

    def __init__(self) -> None:
        self._latency_buf: deque[float] = deque()   # raw ms values, unbounded
        self._lock = threading.Lock()

        # Cache for the most recently computed values (populated on each call)
        self._latest_recall_at_k: MetricResult | None = None
        self._latest_mrr: MetricResult | None = None
        self._latest_citation_accuracy: MetricResult | None = None
        self._latest_hallucination_rate: MetricResult | None = None

    # ------------------------------------------------------------------
    # Latency recording (Requirement 9.3)
    # ------------------------------------------------------------------

    def record_latency_ms(self, ms: float) -> None:
        """Append one end-to-end latency sample (in milliseconds) to the buffer.

        Args:
            ms: Measured end-to-end response time in milliseconds.  Must be a
                finite non-negative number; negative values are clamped to 0.
        """
        value = max(0.0, float(ms))
        with self._lock:
            self._latency_buf.append(value)

    # ------------------------------------------------------------------
    # p95 latency (Requirements 10.1, 10.2)
    # ------------------------------------------------------------------

    def p95_latency_s(self) -> MetricResult:
        """Compute the 95th-percentile latency in seconds.

        Returns a :class:`MetricResult` with:
        - ``value`` set to the p95 in seconds (≥2 decimal precision) when the
          buffer holds ≥30 samples.
        - ``is_insufficient=True`` and ``value=None`` when fewer than 30 samples
          are available (Requirement 10.2).

        Returns:
            MetricResult describing p95 or the insufficiency state.
        """
        with self._lock:
            samples = list(self._latency_buf)

        n = len(samples)
        if n < self.MIN_LATENCY_SAMPLES:
            return MetricResult(
                value=None,
                sample_size=n,
                is_insufficient=True,
            )

        # Nearest-rank percentile (standard definition used throughout the
        # design).  Sort a copy so we never mutate the deque order.
        import math
        sorted_samples = sorted(samples)
        # Index for the 95th percentile (1-based nearest-rank: ceil(p/100 * n))
        rank = math.ceil(0.95 * n)
        p95_ms = sorted_samples[rank - 1]
        # Convert to seconds and round to ≥2 decimal places (Req 10.1).
        # Python floats provide ≥15 significant digits, so rounding to 2 dp
        # satisfies the ≥2-decimal-places reporting contract.
        p95_s = round(p95_ms / 1000.0, 2)

        return MetricResult(
            value=p95_s,
            sample_size=n,
        )

    # ------------------------------------------------------------------
    # Retrieval quality: R@k and MRR (Requirement 10.3)
    # ------------------------------------------------------------------

    def recall_at_k_and_mrr(self, eval_set: list[dict[str, Any]]) -> MetricResult:
        """Compute Recall@k and MRR over a labeled evaluation query set.

        Both R@k and MRR are stored internally; only one :class:`MetricResult`
        is returned here (for R@k).  Use :meth:`report` to retrieve both.

        Each entry in *eval_set* must contain:
        - ``"retrieved_chunks"``: list of retrieved chunk ids / objects whose
          string representation is used for matching.
        - ``"relevant_chunk_ids"``: list of labeled relevant chunk ids.

        Args:
            eval_set: List of evaluation query dicts.

        Returns:
            MetricResult for Recall@k.  MRR is cached for :meth:`report`.

        Raises:
            Nothing — all error conditions are encoded in the returned result.
        """
        # Guard: empty or under-labeled set → error (Requirement 10.6)
        validation_error = self._validate_eval_set(
            eval_set,
            required_keys=["retrieved_chunks", "relevant_chunk_ids"],
            metric_name="R@k / MRR",
        )
        if validation_error:
            result = MetricResult(value=None, sample_size=len(eval_set), error=validation_error)
            self._latest_recall_at_k = result
            self._latest_mrr = result
            return result

        n = len(eval_set)
        recall_sum = 0.0
        mrr_sum = 0.0

        for entry in eval_set:
            retrieved = entry["retrieved_chunks"]
            relevant_ids = set(str(r) for r in (entry.get("relevant_chunk_ids") or []))

            if not relevant_ids:
                # Entry has an empty label set — treat recall/rr as 0 for this query
                continue

            # Normalize retrieved chunk ids to strings for comparison
            retrieved_ids = [str(c) for c in retrieved]

            # Recall@k: fraction of relevant chunks present in the retrieved list
            hits = sum(1 for rid in retrieved_ids if rid in relevant_ids)
            recall_sum += hits / len(relevant_ids)

            # MRR: reciprocal rank of the first relevant hit
            rr = 0.0
            for rank, rid in enumerate(retrieved_ids, start=1):
                if rid in relevant_ids:
                    rr = 1.0 / rank
                    break
            mrr_sum += rr

        recall_at_k = recall_sum / n
        mrr = mrr_sum / n

        # Clamp to [0, 1] as a safety measure
        recall_at_k = max(0.0, min(1.0, recall_at_k))
        mrr = max(0.0, min(1.0, mrr))

        recall_result = MetricResult(value=recall_at_k, sample_size=n)
        mrr_result = MetricResult(value=mrr, sample_size=n)

        self._latest_recall_at_k = recall_result
        self._latest_mrr = mrr_result
        return recall_result

    # ------------------------------------------------------------------
    # Citation accuracy (Requirement 10.4)
    # ------------------------------------------------------------------

    def citation_accuracy(self, eval_set: list[dict[str, Any]]) -> MetricResult:
        """Compute citation accuracy over a labeled evaluation query set.

        Citation accuracy = (citations whose cited page is among the reference
        source pages) / (total citations across all queries).

        Each entry in *eval_set* must contain:
        - ``"citations"``: list of :class:`~backend.data_models.Citation`-like
          objects or dicts with a ``"pages"`` attribute/key.
        - ``"reference_source_pages"``: list of labeled ground-truth page
          numbers.

        Args:
            eval_set: List of evaluation query dicts.

        Returns:
            MetricResult for citation accuracy in [0, 1].
        """
        validation_error = self._validate_eval_set(
            eval_set,
            required_keys=["citations", "reference_source_pages"],
            metric_name="citation_accuracy",
        )
        if validation_error:
            result = MetricResult(value=None, sample_size=len(eval_set), error=validation_error)
            self._latest_citation_accuracy = result
            return result

        n = len(eval_set)
        total_citations = 0
        correct_citations = 0

        for entry in eval_set:
            citations = entry.get("citations") or []
            reference_pages = set(int(p) for p in (entry.get("reference_source_pages") or []))

            for citation in citations:
                # Support both Citation dataclass (has .pages) and plain dicts
                if hasattr(citation, "pages"):
                    cited_pages = citation.pages
                elif isinstance(citation, dict):
                    cited_pages = citation.get("pages", [])
                else:
                    cited_pages = []

                for page in cited_pages:
                    total_citations += 1
                    if int(page) in reference_pages:
                        correct_citations += 1

        if total_citations == 0:
            # No citations were generated — define accuracy as 1.0 (nothing wrong)
            accuracy = 1.0
        else:
            accuracy = correct_citations / total_citations

        accuracy = max(0.0, min(1.0, accuracy))
        result = MetricResult(value=accuracy, sample_size=n)
        self._latest_citation_accuracy = result
        return result

    # ------------------------------------------------------------------
    # Hallucination rate (Requirement 10.5)
    # ------------------------------------------------------------------

    def hallucination_rate(self, eval_set: list[dict[str, Any]]) -> MetricResult:
        """Compute hallucination rate over a labeled evaluation query set.

        Hallucination rate = (answers containing ≥1 claim that cannot be matched
        to any retrieved chunk) / (total answers).

        Each entry in *eval_set* must contain:
        - ``"generated_answer"``: the answer string produced by the generator.
        - ``"verified_claims"``: list of claim strings that are verifiable from
          the retrieved chunks.  An answer is *hallucinated* if it contains at
          least one sentence/claim not present in this list.

        For simplicity the check is: the answer text is split into individual
        claims/sentences and those not found anywhere in the ``verified_claims``
        list (substring match) are flagged as unverified.

        Args:
            eval_set: List of evaluation query dicts.

        Returns:
            MetricResult for hallucination rate in [0, 1].
        """
        validation_error = self._validate_eval_set(
            eval_set,
            required_keys=["generated_answer", "verified_claims"],
            metric_name="hallucination_rate",
        )
        if validation_error:
            result = MetricResult(value=None, sample_size=len(eval_set), error=validation_error)
            self._latest_hallucination_rate = result
            return result

        n = len(eval_set)
        hallucinated_count = 0

        for entry in eval_set:
            answer = entry.get("generated_answer") or ""
            verified_claims = entry.get("verified_claims") or []

            # Split the answer into simple claim units (sentences).
            claims = self._split_into_claims(answer)

            answer_has_hallucination = False
            for claim in claims:
                claim_stripped = claim.strip()
                if not claim_stripped:
                    continue
                # A claim is considered verified if it appears (as a substring)
                # in any of the verified_claims strings.
                is_verified = any(
                    claim_stripped.lower() in vc.lower()
                    for vc in verified_claims
                )
                if not is_verified:
                    answer_has_hallucination = True
                    break

            if answer_has_hallucination:
                hallucinated_count += 1

        rate = hallucinated_count / n if n > 0 else 0.0
        rate = max(0.0, min(1.0, rate))
        result = MetricResult(value=rate, sample_size=n)
        self._latest_hallucination_rate = result
        return result

    # ------------------------------------------------------------------
    # Report (Requirement 10.7)
    # ------------------------------------------------------------------

    def report(self) -> MetricsReport:
        """Return the latest computed metric values, each annotated with the
        sample size over which it was computed.

        The p95 latency is computed fresh from the current ring buffer on every
        call.  Retrieval and answer metrics use the most recently cached values
        (i.e. the last time each corresponding method was called).  If a metric
        has never been computed, its field is ``None``.

        Returns:
            MetricsReport snapshot.
        """
        return MetricsReport(
            p95_latency=self.p95_latency_s(),
            recall_at_k=self._latest_recall_at_k,
            mrr=self._latest_mrr,
            citation_accuracy=self._latest_citation_accuracy,
            hallucination_rate=self._latest_hallucination_rate,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_eval_set(
        self,
        eval_set: list[dict[str, Any]],
        required_keys: list[str],
        metric_name: str,
    ) -> str | None:
        """Return an error string if the eval set is unusable, else ``None``.

        Checks performed (Requirement 10.6):
        1. The set must not be empty.
        2. The set must have at least MIN_EVAL_QUERIES entries.
        3. Every entry must contain non-None values for all required_keys.

        Args:
            eval_set:      The evaluation query list to validate.
            required_keys: Keys that each entry must carry for this metric.
            metric_name:   Human-readable metric name used in error messages.

        Returns:
            An error string, or ``None`` if the set is valid.
        """
        if not eval_set:
            return (
                f"Evaluation set is empty; cannot compute {metric_name}. "
                "Provide at least one labeled query."
            )

        n = len(eval_set)
        if n < self.MIN_EVAL_QUERIES:
            return (
                f"Evaluation set has only {n} quer{'y' if n == 1 else 'ies'}; "
                f"at least {self.MIN_EVAL_QUERIES} are required to compute "
                f"{metric_name}."
            )

        # Check that every entry has the required keys with non-None values
        missing_info: list[str] = []
        for i, entry in enumerate(eval_set):
            for key in required_keys:
                if key not in entry or entry[key] is None:
                    missing_info.append(f"entry[{i}] is missing required key '{key}'")

        if missing_info:
            summary = "; ".join(missing_info[:5])  # cap to first 5 for brevity
            if len(missing_info) > 5:
                summary += f" (and {len(missing_info) - 5} more)"
            return (
                f"Evaluation set has missing labels for {metric_name}: {summary}. "
                "All entries must include the required fields."
            )

        return None

    @staticmethod
    def _split_into_claims(text: str) -> list[str]:
        """Split answer text into individual claim units for hallucination checking.

        Uses simple sentence-boundary splitting on ``. ``, ``! ``, and ``? ``.

        Args:
            text: Answer text to split.

        Returns:
            List of claim strings (may include empty strings; callers should
            strip and skip empties).
        """
        import re
        # Split on sentence-ending punctuation followed by a space or end-of-string
        parts = re.split(r"(?<=[.!?])\s+", text)
        return parts
