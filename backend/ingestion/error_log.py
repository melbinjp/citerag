"""
backend/ingestion/error_log.py

Structured ingestion error log — JSON Lines format.

Each entry is a self-contained JSON object on a single line:
    {"timestamp": "...", "filename": "...", "page_number": 42,
     "stage": "ocr", "reason": "tesseract returned no text"}

Requirements: 13.1, 13.2, 13.3, 13.5
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class IngestionErrorEntry:
    """One structured failure record emitted by the Ingestion_Pipeline.

    Fields
    ------
    filename    : str              — PDF filename (basename, not full path)
    page_number : int | None       — 1-based page number; None when the failure
                                     is document-level (e.g. corrupt PDF)
    stage       : str              — one of "extraction", "ocr", "embedding",
                                     "persistence"
    reason      : str              — human-readable description of the failure
    timestamp   : str              — ISO 8601 timestamp (UTC, e.g.
                                     "2025-01-01T12:00:00Z")
    """

    filename: str
    page_number: Optional[int]
    stage: str
    reason: str
    timestamp: str

    # ------------------------------------------------------------------
    # Convenience constructor
    # ------------------------------------------------------------------

    @classmethod
    def now(
        cls,
        filename: str,
        stage: str,
        reason: str,
        page_number: Optional[int] = None,
    ) -> "IngestionErrorEntry":
        """Create an entry with the current UTC time as the timestamp."""
        ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return cls(
            filename=filename,
            page_number=page_number,
            stage=stage,
            reason=reason,
            timestamp=ts,
        )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a plain dict suitable for JSON serialisation."""
        return asdict(self)

    def to_json_line(self) -> str:
        """Serialise to a single JSON object with no trailing newline."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "IngestionErrorEntry":
        """Reconstruct an entry from a plain dict (e.g. parsed from JSON)."""
        return cls(
            filename=data["filename"],
            page_number=data.get("page_number"),  # may be null / absent
            stage=data["stage"],
            reason=data["reason"],
            timestamp=data["timestamp"],
        )

    @classmethod
    def from_json_line(cls, line: str) -> "IngestionErrorEntry":
        """Reconstruct an entry from a single JSON-Lines text line."""
        return cls.from_dict(json.loads(line))


# ---------------------------------------------------------------------------
# Log class
# ---------------------------------------------------------------------------

class IngestionErrorLog:
    """In-memory buffer for ingestion error entries with JSON Lines I/O.

    Typical usage
    -------------
        log = IngestionErrorLog()
        log.append(IngestionErrorEntry.now("report.pdf", "ocr",
                                           "tesseract returned no text", 42))
        log.flush("/data/ingestion_errors.jsonl")   # Req 13.5

    Reading back after a run
    ------------------------
        entries = IngestionErrorLog.read("/data/ingestion_errors.jsonl")
    """

    def __init__(self) -> None:
        self._entries: list[IngestionErrorEntry] = []

    # ------------------------------------------------------------------
    # Write side
    # ------------------------------------------------------------------

    def append(self, entry: IngestionErrorEntry) -> None:
        """Add *entry* to the in-memory buffer (Req 13.1, 13.2)."""
        self._entries.append(entry)

    @property
    def entries(self) -> list[IngestionErrorEntry]:
        """Return a snapshot of the current in-memory buffer (read-only copy)."""
        return list(self._entries)

    def flush(self, path: str) -> None:
        """Write all buffered entries to *path* as JSON Lines (Req 13.3, 13.5).

        - The parent directory is created if it does not exist.
        - Entries are *appended* to the file so that multiple partial runs
          accumulate in one log without losing earlier failures.
        - Each entry occupies exactly one line terminated by '\\n'.
        - The file is flushed and synced to durable storage before returning.
        """
        if not self._entries:
            return

        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(path, "a", encoding="utf-8") as fh:
            for entry in self._entries:
                fh.write(entry.to_json_line())
                fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())

    # ------------------------------------------------------------------
    # Read side
    # ------------------------------------------------------------------

    @staticmethod
    def read(path: str) -> list[IngestionErrorEntry]:
        """Parse a JSON Lines file at *path* and return all entries (Req 13.3).

        - Blank lines are skipped silently.
        - Raises ``FileNotFoundError`` when *path* does not exist.
        - Raises ``json.JSONDecodeError`` or ``KeyError`` on malformed lines.
        """
        entries: list[IngestionErrorEntry] = []
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                entries.append(IngestionErrorEntry.from_json_line(line))
        return entries

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:  # pragma: no cover
        return f"IngestionErrorLog(buffered={len(self._entries)})"
