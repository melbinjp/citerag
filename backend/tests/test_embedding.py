"""Unit tests for backend/embedding.py.

Tests use mocking to avoid downloading model weights.

Requirements: 4.1, 4.2
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers – build a minimal FlagEmbedding stub that can be imported
# ---------------------------------------------------------------------------

def _make_flag_embedding_stub():
    """Register a minimal FlagEmbedding package in sys.modules so that
    ``from FlagEmbedding import BGEM3FlagModel`` doesn't fail even when
    the real package is not installed in the test environment.
    """
    module = types.ModuleType("FlagEmbedding")
    module.BGEM3FlagModel = MagicMock()  # type: ignore[attr-defined]
    sys.modules.setdefault("FlagEmbedding", module)
    return module


_FLAG_EMBEDDING_STUB = _make_flag_embedding_stub()


# Import after the stub is registered
from backend.embedding import EmbeddingModel, EmbeddingResult, SparseVector  # noqa: E402


# ---------------------------------------------------------------------------
# Helper to build a fake model whose .encode() returns a controlled output
# ---------------------------------------------------------------------------

def _fake_model(n: int, dense_dim: int = 1024):
    """Return a mock BGEM3FlagModel that yields deterministic outputs for *n* texts."""
    model = MagicMock()
    model.encode.return_value = {
        "dense_vecs": np.random.default_rng(42).random((n, dense_dim)).astype(np.float32),
        "lexical_weights": [
            {i * 10: float(i + 1) * 0.1 for i in range(3)} for _ in range(n)
        ],
    }
    return model


# ---------------------------------------------------------------------------
# SparseVector tests
# ---------------------------------------------------------------------------

class TestSparseVector:
    def test_valid_construction(self):
        sv = SparseVector(indices=[1, 2, 3], values=[0.1, 0.2, 0.3])
        assert sv.indices == [1, 2, 3]
        assert sv.values == [0.1, 0.2, 0.3]

    def test_empty_is_valid(self):
        sv = SparseVector(indices=[], values=[])
        assert sv.indices == []
        assert sv.values == []

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            SparseVector(indices=[1, 2], values=[0.1])

    def test_frozen(self):
        sv = SparseVector(indices=[1], values=[0.5])
        with pytest.raises(Exception):  # FrozenInstanceError
            sv.indices = [2]  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EmbeddingResult tests
# ---------------------------------------------------------------------------

class TestEmbeddingResult:
    def test_construction(self):
        sv = SparseVector(indices=[5], values=[0.9])
        er = EmbeddingResult(dense=[0.1] * 1024, sparse=sv)
        assert len(er.dense) == 1024
        assert er.sparse is sv

    def test_frozen(self):
        sv = SparseVector(indices=[], values=[])
        er = EmbeddingResult(dense=[], sparse=sv)
        with pytest.raises(Exception):
            er.dense = []  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EmbeddingModel tests
# ---------------------------------------------------------------------------

class TestEmbeddingModelInit:
    def test_loads_model_once(self):
        fake_model = _fake_model(0)
        with patch.object(EmbeddingModel, "_load_model", return_value=fake_model) as mock_load:
            em = EmbeddingModel("BAAI/bge-m3")
            mock_load.assert_called_once_with("BAAI/bge-m3")
            assert em._model is fake_model

    def test_custom_model_name(self):
        fake_model = _fake_model(0)
        with patch.object(EmbeddingModel, "_load_model", return_value=fake_model):
            em = EmbeddingModel("some-other-model")
            assert em._model_name == "some-other-model"


class TestEmbedEmptyInput:
    def test_empty_list_returns_empty(self):
        fake_model = _fake_model(0)
        with patch.object(EmbeddingModel, "_load_model", return_value=fake_model):
            em = EmbeddingModel()
        result = em.embed([])
        assert result == []
        # Model should NOT have been called
        fake_model.encode.assert_not_called()


class TestEmbedSingleText:
    def test_returns_one_result(self):
        n = 1
        fake_model = _fake_model(n)
        with patch.object(EmbeddingModel, "_load_model", return_value=fake_model):
            em = EmbeddingModel()
        results = em.embed(["hello world"])
        assert len(results) == 1

    def test_dense_vector_length(self):
        n = 1
        fake_model = _fake_model(n, dense_dim=1024)
        with patch.object(EmbeddingModel, "_load_model", return_value=fake_model):
            em = EmbeddingModel()
        result = em.embed(["hello world"])[0]
        assert len(result.dense) == 1024

    def test_dense_values_are_floats(self):
        fake_model = _fake_model(1)
        with patch.object(EmbeddingModel, "_load_model", return_value=fake_model):
            em = EmbeddingModel()
        result = em.embed(["test"])[0]
        assert all(isinstance(v, float) for v in result.dense)

    def test_sparse_indices_and_values(self):
        fake_model = _fake_model(1)
        with patch.object(EmbeddingModel, "_load_model", return_value=fake_model):
            em = EmbeddingModel()
        result = em.embed(["test"])[0]
        sv = result.sparse
        assert isinstance(sv, SparseVector)
        assert len(sv.indices) == len(sv.values)
        assert all(isinstance(i, int) for i in sv.indices)
        assert all(isinstance(v, float) for v in sv.values)


class TestEmbedMultipleTexts:
    def test_returns_correct_count(self):
        texts = ["text one", "text two", "text three", "text four"]
        n = len(texts)
        fake_model = _fake_model(n)
        with patch.object(EmbeddingModel, "_load_model", return_value=fake_model):
            em = EmbeddingModel()
        results = em.embed(texts)
        assert len(results) == n

    def test_order_preserved(self):
        """Each EmbeddingResult corresponds to its input text by position."""
        n = 3
        dense_dim = 4  # small for speed
        rng = np.random.default_rng(0)
        dense_array = rng.random((n, dense_dim)).astype(np.float32)
        sparse_list = [{i: float(i + idx) for i in range(2)} for idx in range(n)]

        fake_model = MagicMock()
        fake_model.encode.return_value = {
            "dense_vecs": dense_array,
            "lexical_weights": sparse_list,
        }

        with patch.object(EmbeddingModel, "_load_model", return_value=fake_model):
            em = EmbeddingModel()
        results = em.embed(["a", "b", "c"])

        # Check each result matches the corresponding row from dense_array
        for idx, res in enumerate(results):
            expected_dense = [float(x) for x in dense_array[idx]]
            assert res.dense == pytest.approx(expected_dense)

    def test_single_forward_pass(self):
        """embed() must call model.encode exactly once regardless of list size."""
        n = 5
        fake_model = _fake_model(n)
        with patch.object(EmbeddingModel, "_load_model", return_value=fake_model):
            em = EmbeddingModel()
        em.embed(["t1", "t2", "t3", "t4", "t5"])
        fake_model.encode.assert_called_once()

    def test_encode_called_with_correct_flags(self):
        n = 2
        fake_model = _fake_model(n)
        with patch.object(EmbeddingModel, "_load_model", return_value=fake_model):
            em = EmbeddingModel()
        em.embed(["hello", "world"])
        _, kwargs = fake_model.encode.call_args
        assert kwargs.get("return_dense") is True
        assert kwargs.get("return_sparse") is True
        assert kwargs.get("return_colbert_vecs") is False

    def test_all_results_are_embedding_result_instances(self):
        n = 4
        fake_model = _fake_model(n)
        with patch.object(EmbeddingModel, "_load_model", return_value=fake_model):
            em = EmbeddingModel()
        results = em.embed(["a", "b", "c", "d"])
        for r in results:
            assert isinstance(r, EmbeddingResult)


class TestEmbedOutputValidation:
    def test_dense_count_mismatch_raises(self):
        """If model returns wrong number of dense vectors, raise ValueError."""
        n_input = 3
        fake_model = MagicMock()
        fake_model.encode.return_value = {
            "dense_vecs": np.zeros((2, 1024), dtype=np.float32),  # wrong count
            "lexical_weights": [{1: 0.5} for _ in range(n_input)],
        }
        with patch.object(EmbeddingModel, "_load_model", return_value=fake_model):
            em = EmbeddingModel()
        with pytest.raises(ValueError, match="dense vectors"):
            em.embed(["a", "b", "c"])

    def test_sparse_count_mismatch_raises(self):
        """If model returns wrong number of sparse vectors, raise ValueError."""
        n_input = 3
        fake_model = MagicMock()
        fake_model.encode.return_value = {
            "dense_vecs": np.zeros((n_input, 1024), dtype=np.float32),
            "lexical_weights": [{1: 0.5}],  # only 1 instead of 3
        }
        with patch.object(EmbeddingModel, "_load_model", return_value=fake_model):
            em = EmbeddingModel()
        with pytest.raises(ValueError, match="sparse vectors"):
            em.embed(["a", "b", "c"])


class TestExtractSparse:
    def test_sorted_indices(self):
        """Sparse indices should be sorted by term-id for determinism."""
        raw = {30: 0.3, 10: 0.1, 20: 0.2}
        sv = EmbeddingModel._extract_sparse(raw)
        assert sv.indices == [10, 20, 30]
        assert sv.values == pytest.approx([0.1, 0.2, 0.3])

    def test_empty_dict(self):
        sv = EmbeddingModel._extract_sparse({})
        assert sv.indices == []
        assert sv.values == []

    def test_non_dict_raises(self):
        with pytest.raises(TypeError, match="Unexpected sparse output type"):
            EmbeddingModel._extract_sparse([1, 2, 3])  # type: ignore[arg-type]


class TestMissingFlagEmbedding:
    def test_import_error_raised_with_helpful_message(self):
        """If FlagEmbedding is not installed, a clear ImportError is raised."""
        with patch.dict(sys.modules, {"FlagEmbedding": None}):
            with pytest.raises(ImportError, match="FlagEmbedding"):
                EmbeddingModel._load_model("BAAI/bge-m3")
