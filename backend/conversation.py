"""
Conversation session store for multi-turn conversation history.

Maintains per-session ordered turn history with bounds:
- At most MAX_TURNS (10) turns
- At most MAX_TOKENS (4000) tokens (word-level approximation)

When a new turn would exceed either bound AND prior history exists, oldest turns
are evicted until both bounds are satisfied (Req 12.4).
When the session is empty before append, the eviction pass is skipped entirely
as a fast path (Req 12.5).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    """A single conversation turn consisting of a question and its answer."""

    question: str
    answer: str


def _token_count(turn: Turn) -> int:
    """Count tokens for a turn using word-level approximation."""
    return len(turn.question.split()) + len(turn.answer.split())


def _total_tokens(turns: list[Turn]) -> int:
    """Sum token counts across all turns in the list."""
    return sum(_token_count(t) for t in turns)


class ConversationStore:
    """Per-session in-memory conversation history store.

    Requirements: 12.1, 12.4, 12.5, 12.6, 12.7
    """

    MAX_TURNS = 10
    MAX_TOKENS = 4000

    def __init__(self) -> None:
        # Internal storage: session_id -> list[Turn] (ordered, oldest first)
        self._store: dict[str, list[Turn]] = {}

    def append(self, session_id: str, turn: Turn) -> None:
        """Append a turn to the session history.

        When the session already holds prior turns, evict the oldest turns
        until both the 10-turn and 4000-token bounds hold.

        When the session is empty before this append (no prior turns), skip
        the history-cleanup/eviction pass entirely as a fast path (Req 12.5).

        Args:
            session_id: Unique session identifier.
            turn: The Turn (question + answer) to append.
        """
        existing = self._store.get(session_id)

        if existing is None:
            # First interaction with this session_id — initialise and fast-path
            self._store[session_id] = [turn]
            return

        if len(existing) == 0:
            # Session was explicitly created/cleared — fast path: no eviction
            existing.append(turn)
            return

        # Session already has prior turns: evict oldest until bounds are met
        # Append the new turn first so we measure combined size.
        existing.append(turn)

        # Evict oldest turns until both constraints are satisfied
        while len(existing) > self.MAX_TURNS or _total_tokens(existing) > self.MAX_TOKENS:
            if len(existing) == 1:
                # Only the just-appended turn remains; cannot evict further.
                break
            existing.pop(0)

    def history(self, session_id: str) -> list[Turn]:
        """Return a copy of the turn list for the session.

        Returns an empty list for unknown/cleared sessions.

        Args:
            session_id: Unique session identifier.

        Returns:
            A shallow copy of the ordered list of turns (oldest first).
        """
        return list(self._store.get(session_id, []))

    def clear(self, session_id: str) -> None:
        """Clear all history for the session (Req 12.6).

        After clearing, the session entry is removed so subsequent appends
        use the fast-path for empty sessions.

        Args:
            session_id: Unique session identifier.
        """
        self._store.pop(session_id, None)
