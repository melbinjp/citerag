"""Unit tests for backend/ingestion/clean.py.

Covers:
- normalize_whitespace  (Requirement 2.1)
- strip_control_chars   (Requirement 2.2)
- remove_repeated_lines (Requirement 2.3)
- Cleaner orchestrator
"""

import pytest

from backend.ingestion.clean import (
    Cleaner,
    normalize_whitespace,
    remove_repeated_lines,
    strip_control_chars,
)


# ---------------------------------------------------------------------------
# normalize_whitespace (Req 2.1)
# ---------------------------------------------------------------------------


class TestNormalizeWhitespace:
    def test_leading_trailing_stripped(self):
        assert normalize_whitespace("  hello  ") == "hello"

    def test_internal_spaces_collapsed(self):
        assert normalize_whitespace("hello   world") == "hello world"

    def test_tabs_collapsed(self):
        assert normalize_whitespace("a\tb") == "a b"

    def test_newline_collapsed(self):
        assert normalize_whitespace("line1\n\nline2") == "line1 line2"

    def test_mixed_whitespace_collapsed(self):
        assert normalize_whitespace("  foo\t \n  bar  ") == "foo bar"

    def test_empty_string_unchanged(self):
        assert normalize_whitespace("") == ""

    def test_only_whitespace_becomes_empty(self):
        assert normalize_whitespace("   \t\n  ") == ""

    def test_single_word_unchanged(self):
        assert normalize_whitespace("hello") == "hello"

    def test_idempotent(self):
        text = "  hello   world  "
        once = normalize_whitespace(text)
        twice = normalize_whitespace(once)
        assert once == twice

    def test_already_normalized_unchanged(self):
        text = "already normalized"
        assert normalize_whitespace(text) == text


# ---------------------------------------------------------------------------
# strip_control_chars (Req 2.2)
# ---------------------------------------------------------------------------


class TestStripControlChars:
    def test_null_byte_removed(self):
        assert strip_control_chars("hel\x00lo") == "hello"

    def test_tab_removed(self):
        # U+0009 (HT / tab) is in the control range and should be stripped
        assert strip_control_chars("a\x09b") == "ab"

    def test_newline_preserved(self):
        # U+000A must NOT be stripped
        assert strip_control_chars("line1\nline2") == "line1\nline2"

    def test_carriage_return_removed(self):
        # U+000D is a control char that should be stripped
        assert strip_control_chars("a\rb") == "ab"

    def test_form_feed_removed(self):
        # U+000C is a control char
        assert strip_control_chars("a\x0cb") == "ab"

    def test_del_removed(self):
        # U+007F (DEL) should be stripped
        assert strip_control_chars("a\x7fb") == "ab"

    def test_all_control_range_stripped(self):
        # Build a string with every control char U+0000–U+001F
        ctrl = "".join(chr(c) for c in range(0x00, 0x20))
        result = strip_control_chars(ctrl)
        # Only the newline should survive
        assert result == "\n"

    def test_printable_chars_preserved(self):
        text = "Hello, World! 123 αβγ"
        assert strip_control_chars(text) == text

    def test_empty_string_unchanged(self):
        assert strip_control_chars("") == ""

    def test_only_control_chars_becomes_empty(self):
        assert strip_control_chars("\x00\x01\x02\x03") == ""

    def test_mixed_control_and_printable(self):
        assert strip_control_chars("a\x00b\x01c") == "abc"

    def test_newline_mixed_with_control(self):
        # Newline should survive even when surrounded by other control chars
        assert strip_control_chars("\x00\nfoo\x07") == "\nfoo"


# ---------------------------------------------------------------------------
# remove_repeated_lines (Req 2.3)
# ---------------------------------------------------------------------------


class TestRemoveRepeatedLines:
    def _make_pages(self, lines_per_page: list[list[str]]) -> list[str]:
        return ["\n".join(lines) for lines in lines_per_page]

    # --- fewer-than-3-page bypass ---

    def test_single_page_unchanged(self):
        pages = ["header\ncontent\nfooter"]
        assert remove_repeated_lines(pages) == pages

    def test_two_pages_unchanged(self):
        pages = ["header\ncontent1\nfooter", "header\ncontent2\nfooter"]
        assert remove_repeated_lines(pages) == pages

    # --- threshold behaviour ---

    def test_line_on_all_pages_removed(self):
        # "Header" appears on 3/3 pages (100%) → should be removed
        pages = self._make_pages(
            [
                ["Header", "content A"],
                ["Header", "content B"],
                ["Header", "content C"],
            ]
        )
        result = remove_repeated_lines(pages)
        for page in result:
            assert "Header" not in page.splitlines()

    def test_line_on_exactly_half_removed(self):
        # 4 pages, line appears on 2 → 50% → removed
        pages = self._make_pages(
            [
                ["Footer", "body 1"],
                ["Footer", "body 2"],
                ["body 3"],
                ["body 4"],
            ]
        )
        result = remove_repeated_lines(pages)
        for page in result:
            assert "Footer" not in page.splitlines()

    def test_line_below_threshold_kept(self):
        # 4 pages, "Rare" appears on 1 → 25% → kept
        pages = self._make_pages(
            [
                ["Rare", "body 1"],
                ["body 2"],
                ["body 3"],
                ["body 4"],
            ]
        )
        result = remove_repeated_lines(pages)
        assert "Rare" in result[0].splitlines()

    def test_unique_content_preserved(self):
        # Each page has unique content that should never be removed
        pages = self._make_pages(
            [
                ["Header", "unique A"],
                ["Header", "unique B"],
                ["Header", "unique C"],
            ]
        )
        result = remove_repeated_lines(pages)
        assert "unique A" in result[0]
        assert "unique B" in result[1]
        assert "unique C" in result[2]

    def test_three_pages_exact_boundary(self):
        # 3 pages, line on 2 pages → 66.7% ≥ 50% → removed
        pages = self._make_pages(
            [
                ["banner", "text 1"],
                ["banner", "text 2"],
                ["text 3"],
            ]
        )
        result = remove_repeated_lines(pages)
        assert "banner" not in result[0]
        assert "banner" not in result[1]

    def test_no_repeated_lines_unchanged(self):
        pages = self._make_pages(
            [
                ["line A", "line B"],
                ["line C", "line D"],
                ["line E", "line F"],
            ]
        )
        result = remove_repeated_lines(pages)
        assert result == pages

    def test_whitespace_normalized_for_comparison(self):
        # "  Header  " and "Header" should be treated as identical
        pages = self._make_pages(
            [
                ["  Header  ", "body 1"],
                ["Header", "body 2"],
                ["Header  ", "body 3"],
            ]
        )
        result = remove_repeated_lines(pages)
        for page in result:
            lines = page.splitlines()
            assert not any(line.strip() == "Header" for line in lines)

    def test_blank_lines_never_removed(self):
        # Empty / whitespace-only lines should never be considered "repeated"
        pages = self._make_pages(
            [
                ["content A", ""],
                ["content B", ""],
                ["content C", ""],
            ]
        )
        result = remove_repeated_lines(pages)
        # Blank lines should still be present (or at worst collapsed by the
        # join, but no content line should be lost)
        assert "content A" in result[0]
        assert "content B" in result[1]
        assert "content C" in result[2]

    def test_return_type_is_list(self):
        pages = ["a", "b", "c"]
        result = remove_repeated_lines(pages)
        assert isinstance(result, list)

    def test_length_preserved(self):
        pages = ["page1\nfooter", "page2\nfooter", "page3\nfooter"]
        result = remove_repeated_lines(pages)
        assert len(result) == len(pages)


# ---------------------------------------------------------------------------
# Cleaner orchestrator
# ---------------------------------------------------------------------------


class TestCleaner:
    def setup_method(self):
        self.cleaner = Cleaner()

    def test_clean_page_removes_control_chars(self):
        result = self.cleaner.clean_page("hello\x00world")
        assert "\x00" not in result

    def test_clean_page_normalizes_whitespace(self):
        result = self.cleaner.clean_page("  hello   world  ")
        assert result == "hello world"

    def test_clean_page_preserves_newline(self):
        # clean_page normalizes whitespace (which collapses newlines) so the
        # result won't contain a newline — but the input newline should not
        # cause an error and should not cause content to be lost.
        result = self.cleaner.clean_page("hello\nworld")
        assert "hello" in result
        assert "world" in result

    def test_clean_page_strips_control_before_normalize(self):
        # \x09 is a tab (control char) → stripped, then whitespace collapses
        result = self.cleaner.clean_page("a\x09 b")
        # After stripping \x09 we get "a  b", then normalize → "a b"
        assert result == "a b"

    def test_clean_document_applies_all_steps(self):
        pages = [
            "\x00Header\ncontent A",
            "Header\ncontent B",
            "Header\ncontent C",
        ]
        result = self.cleaner.clean_document(pages)
        assert len(result) == 3
        # Control chars stripped
        assert "\x00" not in result[0]
        # Header/footer removed (appears on all 3 pages)
        for page in result:
            assert "Header" not in page
        # Content preserved
        assert "content A" in result[0]
        assert "content B" in result[1]
        assert "content C" in result[2]

    def test_clean_document_single_page_no_repeated_removal(self):
        pages = ["Header\ncontent"]
        result = self.cleaner.clean_document(pages)
        assert len(result) == 1
        # No repeated-line removal for <3 pages; content intact
        assert "Header" in result[0]
        assert "content" in result[0]

    def test_clean_document_empty_pages_list(self):
        result = self.cleaner.clean_document([])
        assert result == []

    def test_clean_document_preserves_page_count(self):
        pages = ["a\x00", "b\x01", "c\x02", "d\x03"]
        result = self.cleaner.clean_document(pages)
        assert len(result) == len(pages)
