"""Text cleaning and normalization for the RAG PDF ingestion pipeline.

Implements:
- Requirement 2.1: Whitespace normalization (collapse runs to single space, trim)
- Requirement 2.2: Control-character stripping (U+0000–U+001F and U+007F, except U+000A)
- Requirement 2.3: Repeated header/footer line removal (lines appearing on ≥50% of pages
  in PDFs with ≥3 pages)
"""

import re
from collections import Counter


def normalize_whitespace(text: str) -> str:
    """Collapse every run of consecutive whitespace to a single space and trim.

    Newlines (U+000A) are considered whitespace here and will also be collapsed
    within a run. Leading and trailing whitespace is removed.

    Requirement 2.1: WHEN text is extracted from a PDF page, THE Ingestion_Pipeline
    SHALL replace every sequence of one or more consecutive whitespace characters
    with a single space character and remove leading and trailing whitespace.

    Args:
        text: Raw extracted text from a PDF page.

    Returns:
        Text with all consecutive whitespace collapsed to a single space and
        leading/trailing whitespace stripped.

    Examples:
        >>> normalize_whitespace("  hello   world  ")
        'hello world'
        >>> normalize_whitespace("line1\\n\\nline2")
        'line1 line2'
        >>> normalize_whitespace("")
        ''
    """
    if not text:
        return text
    return re.sub(r"\s+", " ", text).strip()


def strip_control_chars(text: str) -> str:
    """Remove Unicode control characters except line feed (U+000A).

    Strips all characters in the ranges U+0000–U+001F and U+007F from the
    input, while preserving U+000A (newline / line feed) and all other
    printable characters.

    Requirement 2.2: WHEN text is extracted from a PDF page, THE Ingestion_Pipeline
    SHALL remove all Unicode control characters in the ranges U+0000 through U+001F
    and U+007F, except the line feed character (U+000A), from that page's text.

    Args:
        text: Raw text that may contain control characters.

    Returns:
        Text with control characters removed, newlines preserved.

    Examples:
        >>> strip_control_chars("hello\\x00world")
        'helloworld'
        >>> strip_control_chars("line1\\nline2")
        'line1\\nline2'
        >>> strip_control_chars("tab\\there")
        'tabhere'
    """
    if not text:
        return text
    # Match control chars U+0000–U+001F and U+007F, but NOT U+000A (newline)
    return re.sub(r"[\x00-\x09\x0b-\x1f\x7f]", "", text)


def remove_repeated_lines(pages: list[str]) -> list[str]:
    """Remove header/footer lines that appear on ≥50% of pages.

    A "line" is compared after whitespace normalization (collapse runs, trim).
    Only applied when the document has 3 or more pages. Lines that appear on
    fewer than 50% of pages are preserved unchanged.

    Requirement 2.3: WHEN an identical line (after whitespace normalization)
    appears on at least 50% of the pages of a PDF that contains 3 or more pages,
    THE Ingestion_Pipeline SHALL remove every occurrence of that repeated line
    from each page's text.

    Args:
        pages: List of page text strings (one entry per page of the PDF).

    Returns:
        New list of page texts with repeated lines removed. If the PDF has fewer
        than 3 pages the input pages are returned unchanged.

    Notes:
        - Blank lines (empty or whitespace-only after normalization) are never
          considered repeated lines — removing universal blank lines would alter
          the structure of every page, so they are explicitly excluded.
        - The comparison and removal both operate on the whitespace-normalized
          form of each line.
    """
    total_pages = len(pages)

    # Requirement 2.3: only apply to PDFs with ≥3 pages
    if total_pages < 3:
        return list(pages)

    # Normalize every line across all pages and track how many pages each
    # normalized line appears on (count distinct pages, not total occurrences).
    line_page_count: Counter[str] = Counter()
    for page_text in pages:
        # Collect the unique set of normalized lines for this page so that a
        # line appearing multiple times on the same page is still counted once
        # toward the per-page appearance count.
        normalized_lines_on_page: set[str] = set()
        for raw_line in page_text.splitlines():
            norm = normalize_whitespace(raw_line)
            if norm:  # skip blank lines
                normalized_lines_on_page.add(norm)
        line_page_count.update(normalized_lines_on_page)

    # A line is "repeated" (header/footer) if it appears on ≥50% of pages.
    threshold = total_pages * 0.5
    repeated: set[str] = {
        line for line, count in line_page_count.items() if count >= threshold
    }

    if not repeated:
        return list(pages)

    # Rebuild each page with repeated lines removed.
    cleaned_pages: list[str] = []
    for page_text in pages:
        kept_lines: list[str] = []
        for raw_line in page_text.splitlines():
            norm = normalize_whitespace(raw_line)
            if norm not in repeated:
                kept_lines.append(raw_line)
        cleaned_pages.append("\n".join(kept_lines))

    return cleaned_pages


class Cleaner:
    """Orchestrates whitespace normalization, control-char stripping, and
    repeated header/footer line removal for the ingestion pipeline.

    Usage (single page, no header/footer removal)::

        cleaner = Cleaner()
        clean_text = cleaner.clean_page("  hello\\x00 world  ")

    Usage (full document, with header/footer removal)::

        cleaner = Cleaner()
        clean_pages = cleaner.clean_document(["page1 text", "page2 text", ...])
    """

    def clean_page(self, text: str) -> str:
        """Apply control-char stripping and whitespace normalization to a single page.

        Note: header/footer removal requires the full set of pages and is not
        applied here. Use :meth:`clean_document` for a complete pipeline run.

        Args:
            text: Raw extracted text for a single page.

        Returns:
            Cleaned text with control characters stripped and whitespace normalized.
        """
        text = strip_control_chars(text)
        text = normalize_whitespace(text)
        return text

    def clean_document(self, pages: list[str]) -> list[str]:
        """Clean all pages of a document end-to-end.

        Applies the three cleaning steps in order:
        1. Strip control characters from every page (Req 2.2).
        2. Remove repeated header/footer lines across all pages (Req 2.3).
           This step must happen *before* whitespace normalization so that
           per-line structure (newlines) is still intact for line comparison.
        3. Normalize whitespace on every page (Req 2.1).

        Args:
            pages: List of raw page text strings (one entry per PDF page).

        Returns:
            List of cleaned page text strings in the same order.
        """
        # Step 1: strip control characters (preserves newlines / line structure)
        ctrl_stripped = [strip_control_chars(page) for page in pages]

        # Step 2: cross-page header/footer removal (needs intact line structure)
        after_dedup = remove_repeated_lines(ctrl_stripped)

        # Step 3: whitespace normalization per page
        return [normalize_whitespace(page) for page in after_dedup]
