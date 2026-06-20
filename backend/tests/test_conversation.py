"""
Unit tests for ConversationStore (backend/conversation.py).

Covers:
- Basic append and history retrieval
- Per-session isolation (Req 12.7)
- Fast-path for empty sessions (Req 12.5)
- Turn-bound eviction (Req 12.4)
- Token-bound eviction (Req 12.4)
- Clear (Req 12.6)
- Unknown-session history returns empty list
"""

import sys
import os

# Ensure backend package is importable when running from the tests/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from conversation import ConversationStore, Turn, _token_count, _total_tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_turn(q: str = "question", a: str = "answer") -> Turn:
    return Turn(question=q, answer=a)


def make_turn_with_tokens(n_tokens: int) -> Turn:
    """Return a Turn whose total token count equals n_tokens (split evenly)."""
    half = n_tokens // 2
    remainder = n_tokens - half
    q = " ".join(["word"] * half)
    a = " ".join(["word"] * remainder)
    return Turn(question=q, answer=a)


# ---------------------------------------------------------------------------
# _token_count helper tests
# ---------------------------------------------------------------------------

class TestTokenCount:
    def test_single_word_each(self):
        t = Turn(question="hello", answer="world")
        assert _token_count(t) == 2

    def test_empty_strings(self):
        t = Turn(question="", answer="")
        # "".split() == [] so 0 + 0 = 0
        assert _token_count(t) == 0

    def test_multi_word(self):
        t = Turn(question="one two three", answer="four five")
        assert _token_count(t) == 5

    def test_total_tokens_empty_list(self):
        assert _total_tokens([]) == 0

    def test_total_tokens_single(self):
        t = Turn(question="a b", answer="c")
        assert _total_tokens([t]) == 3

    def test_total_tokens_multiple(self):
        turns = [
            Turn(question="a", answer="b"),   # 2
            Turn(question="c d", answer="e"), # 3
        ]
        assert _total_tokens(turns) == 5


# ---------------------------------------------------------------------------
# Basic append and history
# ---------------------------------------------------------------------------

class TestBasicAppendAndHistory:
    def setup_method(self):
        self.store = ConversationStore()

    def test_history_unknown_session_returns_empty(self):
        assert self.store.history("nonexistent") == []

    def test_single_append_creates_history(self):
        turn = make_turn("q1", "a1")
        self.store.append("s1", turn)
        assert self.store.history("s1") == [turn]

    def test_history_returns_copy(self):
        """Mutating the returned list must not affect internal state."""
        self.store.append("s1", make_turn())
        h = self.store.history("s1")
        h.append(make_turn("extra", "extra"))
        assert len(self.store.history("s1")) == 1

    def test_multiple_appends_preserve_order(self):
        turns = [make_turn(f"q{i}", f"a{i}") for i in range(3)]
        for t in turns:
            self.store.append("s1", t)
        assert self.store.history("s1") == turns

    def test_turn_frozen_dataclass_immutable(self):
        t = make_turn()
        with pytest.raises((AttributeError, TypeError)):
            t.question = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Per-session isolation (Req 12.7)
# ---------------------------------------------------------------------------

class TestPerSessionIsolation:
    def setup_method(self):
        self.store = ConversationStore()

    def test_two_sessions_are_independent(self):
        self.store.append("alice", make_turn("qa", "aa"))
        self.store.append("bob", make_turn("qb", "ab"))

        alice_history = self.store.history("alice")
        bob_history = self.store.history("bob")

        assert len(alice_history) == 1
        assert len(bob_history) == 1
        assert alice_history[0].question == "qa"
        assert bob_history[0].question == "qb"

    def test_clear_one_session_leaves_other_intact(self):
        self.store.append("alice", make_turn("qa", "aa"))
        self.store.append("bob", make_turn("qb", "ab"))

        self.store.clear("alice")

        assert self.store.history("alice") == []
        assert len(self.store.history("bob")) == 1

    def test_sessions_do_not_share_list_reference(self):
        """Internal lists for different sessions must be separate objects."""
        for sid in ("s1", "s2"):
            self.store.append(sid, make_turn(sid, sid))
        h1 = self.store.history("s1")
        h2 = self.store.history("s2")
        assert h1 is not h2


# ---------------------------------------------------------------------------
# Fast-path for empty sessions (Req 12.5)
# ---------------------------------------------------------------------------

class TestFastPathEmptySession:
    def setup_method(self):
        self.store = ConversationStore()

    def test_first_turn_appended_without_eviction(self):
        """A single large turn should not be evicted even if it exceeds bounds."""
        # Create a turn that exceeds MAX_TOKENS by itself
        huge_turn = make_turn_with_tokens(ConversationStore.MAX_TOKENS + 500)
        self.store.append("s1", huge_turn)
        assert self.store.history("s1") == [huge_turn]

    def test_first_turn_after_clear_uses_fast_path(self):
        """After clear, the next append should also skip eviction."""
        # Add some history, clear, then add a large turn
        for i in range(5):
            self.store.append("s1", make_turn(f"q{i}", f"a{i}"))

        self.store.clear("s1")
        huge_turn = make_turn_with_tokens(ConversationStore.MAX_TOKENS + 500)
        self.store.append("s1", huge_turn)

        assert self.store.history("s1") == [huge_turn]

    def test_second_turn_triggers_eviction_logic(self):
        """After one turn exists, the next append should run eviction if needed."""
        # Fill session with MAX_TOKENS worth of tokens in one turn
        big_turn = make_turn_with_tokens(ConversationStore.MAX_TOKENS)
        self.store.append("s1", big_turn)

        # Now append another turn — eviction should run and the oldest gets dropped
        new_turn = make_turn_with_tokens(50)
        self.store.append("s1", new_turn)

        history = self.store.history("s1")
        # Total would be MAX_TOKENS + 50 > MAX_TOKENS, so big_turn should be evicted
        total = _total_tokens(history)
        assert total <= ConversationStore.MAX_TOKENS
        # The newest turn should survive
        assert new_turn in history


# ---------------------------------------------------------------------------
# Turn-bound eviction (Req 12.4)
# ---------------------------------------------------------------------------

class TestTurnBoundEviction:
    def setup_method(self):
        self.store = ConversationStore()

    def test_history_never_exceeds_max_turns(self):
        """Appending more than MAX_TURNS turns should evict the oldest."""
        n = ConversationStore.MAX_TURNS + 5
        turns = [make_turn(f"q{i}", f"a{i}") for i in range(n)]
        for t in turns:
            self.store.append("s1", t)

        history = self.store.history("s1")
        assert len(history) <= ConversationStore.MAX_TURNS

    def test_most_recent_turns_are_retained(self):
        """After eviction the retained turns are the most recent ones."""
        n = ConversationStore.MAX_TURNS + 3
        turns = [make_turn(f"q{i}", f"a{i}") for i in range(n)]
        for t in turns:
            self.store.append("s1", t)

        history = self.store.history("s1")
        expected = turns[n - len(history):]
        assert history == expected

    def test_exactly_max_turns_no_eviction(self):
        """Appending exactly MAX_TURNS turns should keep all of them."""
        turns = [make_turn(f"q{i}", f"a{i}") for i in range(ConversationStore.MAX_TURNS)]
        for t in turns:
            self.store.append("s1", t)

        assert len(self.store.history("s1")) == ConversationStore.MAX_TURNS

    def test_oldest_turns_evicted_first(self):
        """Eviction should always remove the oldest (front) turn first."""
        n = ConversationStore.MAX_TURNS + 1
        turns = [make_turn(f"q{i}", f"a{i}") for i in range(n)]
        for t in turns:
            self.store.append("s1", t)

        history = self.store.history("s1")
        # The first turn (q0) should have been evicted
        assert turns[0] not in history
        # The last appended turn should be present
        assert turns[-1] in history


# ---------------------------------------------------------------------------
# Token-bound eviction (Req 12.4)
# ---------------------------------------------------------------------------

class TestTokenBoundEviction:
    def setup_method(self):
        self.store = ConversationStore()

    def test_history_token_count_never_exceeds_max(self):
        """Total tokens in history must never exceed MAX_TOKENS after any append
        when prior turns exist."""
        # Build sessions with smallish turns to trigger token-only eviction
        turn_tokens = 300  # 13 of these = 3900 (ok), 14 = 4200 (exceeds)
        turns = [make_turn_with_tokens(turn_tokens) for _ in range(15)]
        for t in turns:
            self.store.append("s1", t)

        history = self.store.history("s1")
        assert _total_tokens(history) <= ConversationStore.MAX_TOKENS

    def test_eviction_removes_enough_turns(self):
        """Eviction must continue until both bounds are satisfied."""
        # Add 5 small turns, then a very large turn that pushes tokens way over
        for _ in range(5):
            self.store.append("s1", make_turn_with_tokens(100))  # 500 tokens total

        big_turn = make_turn_with_tokens(3600)  # 4100 total → over MAX_TOKENS
        self.store.append("s1", big_turn)

        history = self.store.history("s1")
        assert _total_tokens(history) <= ConversationStore.MAX_TOKENS

    def test_single_oversized_turn_not_evicted_as_only_entry(self):
        """The only remaining turn is never evicted even if it alone exceeds MAX_TOKENS."""
        self.store.append("s1", make_turn_with_tokens(100))

        # Now append a huge turn — after evicting the small one, huge turn is the only one
        huge_turn = make_turn_with_tokens(ConversationStore.MAX_TOKENS + 1000)
        self.store.append("s1", huge_turn)

        history = self.store.history("s1")
        assert len(history) == 1
        assert history[0] == huge_turn


# ---------------------------------------------------------------------------
# Combined turn + token bound
# ---------------------------------------------------------------------------

class TestCombinedBounds:
    def setup_method(self):
        self.store = ConversationStore()

    def test_both_bounds_enforced_simultaneously(self):
        """After appending many small turns, both MAX_TURNS and MAX_TOKENS must hold."""
        # 200 turns, each 5 tokens → token limit would allow 800 turns, but turn
        # limit caps at 10
        for i in range(200):
            self.store.append("s1", make_turn_with_tokens(5))

        history = self.store.history("s1")
        assert len(history) <= ConversationStore.MAX_TURNS
        assert _total_tokens(history) <= ConversationStore.MAX_TOKENS

    def test_token_bound_kicks_in_before_turn_bound(self):
        """Large turns should trigger token-based eviction before turn limit is hit."""
        # Each turn is 1000 tokens; after 4 turns (4000 tokens) is the limit
        # A 5th turn (5000 > 4000) should evict the oldest
        for i in range(4):
            self.store.append("s1", make_turn_with_tokens(1000))

        assert _total_tokens(self.store.history("s1")) == 4000

        self.store.append("s1", make_turn_with_tokens(1000))

        history = self.store.history("s1")
        assert len(history) == 4  # turn limit not hit yet; one evicted for tokens
        assert _total_tokens(history) <= ConversationStore.MAX_TOKENS


# ---------------------------------------------------------------------------
# Clear (Req 12.6)
# ---------------------------------------------------------------------------

class TestClear:
    def setup_method(self):
        self.store = ConversationStore()

    def test_clear_empties_session(self):
        for i in range(3):
            self.store.append("s1", make_turn(f"q{i}", f"a{i}"))

        self.store.clear("s1")
        assert self.store.history("s1") == []

    def test_clear_nonexistent_session_is_noop(self):
        """Clearing a session that never existed should not raise."""
        self.store.clear("ghost")  # must not raise
        assert self.store.history("ghost") == []

    def test_clear_allows_fresh_start(self):
        """After clear, new appends start fresh with no prior context."""
        for i in range(ConversationStore.MAX_TURNS):
            self.store.append("s1", make_turn(f"q{i}", f"a{i}"))

        self.store.clear("s1")

        new_turn = make_turn("fresh_q", "fresh_a")
        self.store.append("s1", new_turn)

        assert self.store.history("s1") == [new_turn]

    def test_clear_twice_is_idempotent(self):
        self.store.append("s1", make_turn())
        self.store.clear("s1")
        self.store.clear("s1")  # second clear must not raise
        assert self.store.history("s1") == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def setup_method(self):
        self.store = ConversationStore()

    def test_empty_question_and_answer(self):
        t = Turn(question="", answer="")
        self.store.append("s1", t)
        assert self.store.history("s1") == [t]

    def test_multiple_sessions_many_turns(self):
        for sid in ("alice", "bob", "carol"):
            for i in range(15):
                self.store.append(sid, make_turn(f"{sid}_q{i}", f"{sid}_a{i}"))

        for sid in ("alice", "bob", "carol"):
            h = self.store.history(sid)
            assert len(h) <= ConversationStore.MAX_TURNS
            assert _total_tokens(h) <= ConversationStore.MAX_TOKENS

    def test_constants_correct(self):
        assert ConversationStore.MAX_TURNS == 10
        assert ConversationStore.MAX_TOKENS == 4000
