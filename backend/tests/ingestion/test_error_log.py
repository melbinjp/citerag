"""
Unit tests for backend/ingestion/error_log.py

Covers:
  - IngestionErrorEntry construction, serialisation, and round-trip (Req 13.2, 13.3)
  - IngestionErrorLog.append / .entries / .flush / .read lifecycle (Req 13.1, 13.5)
  - Edge cases: blank lines, append-mode accumulation, missing page_number
"""

from __future__ import annotations

import json
import os
import re

import pytest

from backend.ingestion.error_log import IngestionErrorEntry, IngestionErrorLog

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_STAGES = ("extraction", "ocr", "embedding", "persistence")
ISO8601_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def make_entry(
    filename: str = "doc.pdf",
    stage: str = "ocr",
    reason: str = "tesseract returned no text",
    page_number: int | None = 1,
    timestamp: str = "2025-01-01T12:00:00Z",
) -> IngestionErrorEntry:
    return IngestionErrorEntry(
        filename=filename,
        page_number=page_number,
        stage=stage,
        reason=reason,
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# IngestionErrorEntry tests
# ---------------------------------------------------------------------------


class TestIngestionErrorEntry:
    def test_fields_stored_correctly(self):
        e = make_entry(page_number=42)
        assert e.filename == "doc.pdf"
        assert e.page_number == 42
        assert e.stage == "ocr"
        assert e.reason == "tesseract returned no text"
        assert e.timestamp == "2025-01-01T12:00:00Z"

    def test_page_number_can_be_none(self):
        e = make_entry(page_number=None)
        assert e.page_number is None

    def test_to_dict_contains_all_keys(self):
        e = make_entry(page_number=5)
        d = e.to_dict()
        for key in ("filename", "page_number", "stage", "reason", "timestamp"):
            assert key in d

    def test_to_json_line_is_valid_json(self):
        e = make_entry()
        line = e.to_json_line()
        parsed = json.loads(line)
        assert isinstance(parsed, dict)

    def test_to_json_line_contains_no_newline(self):
        e = make_entry()
        assert "\n" not in e.to_json_line()

    def test_round_trip_via_json_line(self):
        e = make_entry(page_number=42)
        restored = IngestionErrorEntry.from_json_line(e.to_json_line())
        assert restored == e

    def test_round_trip_with_null_page_number(self):
        e = make_entry(page_number=None)
        restored = IngestionErrorEntry.from_json_line(e.to_json_line())
        assert restored.page_number is None

    def test_from_dict_ignores_absent_page_number(self):
        d = {
            "filename": "x.pdf",
            "stage": "embedding",
            "reason": "oom",
            "timestamp": "2025-06-01T00:00:00Z",
        }
        e = IngestionErrorEntry.from_dict(d)
        assert e.page_number is None

    def test_now_produces_valid_timestamp(self):
        e = IngestionErrorEntry.now("f.pdf", "extraction", "failed")
        assert ISO8601_RE.match(e.timestamp), f"Bad timestamp: {e.timestamp!r}"

    def test_now_uses_supplied_page_number(self):
        e = IngestionErrorEntry.now("f.pdf", "ocr", "empty", page_number=7)
        assert e.page_number == 7

    def test_example_from_design_doc(self):
        """The exact example from the design doc must round-trip identically."""
        raw = '{"timestamp":"2025-01-01T12:00:00Z","filename":"report.pdf","page_number":42,"stage":"ocr","reason":"tesseract returned no text"}'
        entry = IngestionErrorEntry.from_json_line(raw)
        assert entry.filename == "report.pdf"
        assert entry.page_number == 42
        assert entry.stage == "ocr"
        assert entry.reason == "tesseract returned no text"
        assert entry.timestamp == "2025-01-01T12:00:00Z"
        # Re-serialise and compare key-value pairs (key order may differ)
        assert json.loads(entry.to_json_line()) == json.loads(raw)


# ---------------------------------------------------------------------------
# IngestionErrorLog tests
# ---------------------------------------------------------------------------


class TestIngestionErrorLog:
    def test_empty_log_has_zero_length(self):
        log = IngestionErrorLog()
        assert len(log) == 0
        assert log.entries == []

    def test_append_increases_length(self):
        log = IngestionErrorLog()
        log.append(make_entry())
        assert len(log) == 1

    def test_entries_returns_snapshot(self):
        log = IngestionErrorLog()
        e = make_entry()
        log.append(e)
        snapshot = log.entries
        assert snapshot == [e]
        # Mutating the returned list must not affect the internal buffer
        snapshot.clear()
        assert len(log) == 1

    def test_flush_creates_file(self, tmp_path):
        path = str(tmp_path / "errors.jsonl")
        log = IngestionErrorLog()
        log.append(make_entry())
        log.flush(path)
        assert os.path.exists(path)

    def test_flush_writes_one_line_per_entry(self, tmp_path):
        path = str(tmp_path / "errors.jsonl")
        log = IngestionErrorLog()
        for i in range(3):
            log.append(make_entry(filename=f"doc{i}.pdf", page_number=i + 1))
        log.flush(path)
        with open(path, encoding="utf-8") as fh:
            lines = [l for l in fh.readlines() if l.strip()]
        assert len(lines) == 3

    def test_flush_appends_across_calls(self, tmp_path):
        path = str(tmp_path / "errors.jsonl")
        log1 = IngestionErrorLog()
        log1.append(make_entry(filename="a.pdf"))
        log1.flush(path)

        log2 = IngestionErrorLog()
        log2.append(make_entry(filename="b.pdf"))
        log2.flush(path)

        entries = IngestionErrorLog.read(path)
        assert len(entries) == 2
        filenames = {e.filename for e in entries}
        assert filenames == {"a.pdf", "b.pdf"}

    def test_flush_empty_log_does_not_create_file(self, tmp_path):
        path = str(tmp_path / "errors.jsonl")
        log = IngestionErrorLog()
        log.flush(path)
        assert not os.path.exists(path)

    def test_flush_creates_parent_directories(self, tmp_path):
        path = str(tmp_path / "deep" / "nested" / "errors.jsonl")
        log = IngestionErrorLog()
        log.append(make_entry())
        log.flush(path)
        assert os.path.exists(path)

    def test_read_round_trips_all_fields(self, tmp_path):
        path = str(tmp_path / "errors.jsonl")
        original = [
            make_entry(filename="a.pdf", stage="extraction", page_number=None),
            make_entry(filename="b.pdf", stage="ocr", page_number=7),
            make_entry(filename="c.pdf", stage="embedding", page_number=100),
            make_entry(filename="d.pdf", stage="persistence", page_number=1),
        ]
        log = IngestionErrorLog()
        for e in original:
            log.append(e)
        log.flush(path)

        restored = IngestionErrorLog.read(path)
        assert restored == original

    def test_read_skips_blank_lines(self, tmp_path):
        path = str(tmp_path / "errors.jsonl")
        e = make_entry()
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n")
            fh.write(e.to_json_line() + "\n")
            fh.write("   \n")
        entries = IngestionErrorLog.read(path)
        assert len(entries) == 1
        assert entries[0] == e

    def test_read_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            IngestionErrorLog.read(str(tmp_path / "nonexistent.jsonl"))

    def test_each_flushed_line_is_valid_json(self, tmp_path):
        path = str(tmp_path / "errors.jsonl")
        log = IngestionErrorLog()
        log.append(make_entry(reason='reason with "quotes" and \nnewline'))
        log.flush(path)
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    json.loads(line)  # must not raise

    def test_all_stages_serialise(self, tmp_path):
        path = str(tmp_path / "errors.jsonl")
        log = IngestionErrorLog()
        for stage in VALID_STAGES:
            log.append(make_entry(stage=stage))
        log.flush(path)
        entries = IngestionErrorLog.read(path)
        assert [e.stage for e in entries] == list(VALID_STAGES)
