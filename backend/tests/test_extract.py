"""Unit tests for backend/ingestion/extract.py.

Tests cover:
- PageText dataclass (frozen, fields).
- SEMANTIC_IMAGE_PREFIX and MAX_PAGES constants.
- ExtractionError is a proper exception.
- Text_Extractor.extract() logic via mocked fitz:
    - Native text pages → returned as-is (Req 1.1).
    - Zero-native-char pages → routed to full-page OCR (Req 1.2).
    - Pages with native text and embedded images → OCR image, wrap, append (Req 1.3, 1.4).
    - Encrypted PDFs → ExtractionError (Req 1.7).
    - Unopenable PDFs → ExtractionError (Req 1.7).
    - Documents up to MAX_PAGES processed; >MAX_PAGES truncated with warning (Req 1.5).
    - Full-page OCR failure is handled gracefully (empty text, no crash).
    - Embedded-image OCR failure is handled gracefully (no crash).
"""

import sys
import types
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, r"c:\Workspace_Melbin\fun_websites\ackathon\backend")

from ingestion.extract import (
    MAX_PAGES,
    SEMANTIC_IMAGE_PREFIX,
    ExtractionError,
    PageText,
    Text_Extractor,
)


# ──────────────────────────────────────────────────────────────────────────────
# PageText dataclass
# ──────────────────────────────────────────────────────────────────────────────


class TestPageText:
    def test_fields_accessible(self):
        pt = PageText(page_number=3, text="hello world")
        assert pt.page_number == 3
        assert pt.text == "hello world"

    def test_frozen(self):
        pt = PageText(page_number=1, text="a")
        with pytest.raises(FrozenInstanceError):
            pt.page_number = 99  # type: ignore[misc]

    def test_equality_by_value(self):
        assert PageText(1, "x") == PageText(1, "x")
        assert PageText(1, "x") != PageText(2, "x")

    def test_hashable(self):
        """frozen dataclasses must be hashable (usable as dict keys / set members)."""
        s = {PageText(1, "a"), PageText(2, "b")}
        assert len(s) == 2


# ──────────────────────────────────────────────────────────────────────────────
# Module-level constants
# ──────────────────────────────────────────────────────────────────────────────


class TestConstants:
    def test_semantic_prefix(self):
        assert SEMANTIC_IMAGE_PREFIX == "> [System Note: Image Content] "

    def test_max_pages(self):
        assert MAX_PAGES == 2000


# ──────────────────────────────────────────────────────────────────────────────
# ExtractionError
# ──────────────────────────────────────────────────────────────────────────────


class TestExtractionError:
    def test_is_exception(self):
        assert issubclass(ExtractionError, Exception)

    def test_raise_and_catch(self):
        with pytest.raises(ExtractionError, match="something"):
            raise ExtractionError("something went wrong")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: build a fake fitz page / document
# ──────────────────────────────────────────────────────────────────────────────


def _make_fake_page(
    number: int,
    native_text: str,
    images: list | None = None,
    image_rects: dict | None = None,
) -> MagicMock:
    """Return a MagicMock that mimics a fitz.Page.

    Args:
        number: 0-based page index.
        native_text: What page.get_text() returns.
        images: List of image-info tuples (only xref in [0] is used).
        image_rects: Dict mapping xref → list of rect-like objects.
    """
    page = MagicMock()
    page.number = number
    page.get_text.return_value = native_text
    page.get_images.return_value = images or []
    image_rects = image_rects or {}

    def _get_image_rects(xref):
        return image_rects.get(xref, [])

    page.get_image_rects.side_effect = _get_image_rects
    return page


def _make_fake_doc(pages: list[MagicMock], is_encrypted: bool = False) -> MagicMock:
    """Return a MagicMock that mimics a fitz.Document."""
    doc = MagicMock()
    doc.is_encrypted = is_encrypted
    doc.__len__ = MagicMock(return_value=len(pages))
    doc.__getitem__ = MagicMock(side_effect=lambda i: pages[i])

    # extract_image returns a dict with bytes
    doc.extract_image.return_value = {"image": b"\x89PNG\r\n\x1a\n" + b"\x00" * 100}
    doc.close = MagicMock()
    return doc


def _fake_rect(empty: bool = False, infinite: bool = False) -> MagicMock:
    rect = MagicMock()
    rect.is_empty = empty
    rect.is_infinite = infinite
    return rect


# ──────────────────────────────────────────────────────────────────────────────
# Text_Extractor unit tests (fitz and pytesseract mocked)
# ──────────────────────────────────────────────────────────────────────────────


class TestTextExtractorNativeText:
    """Requirement 1.1 – native text extracted and associated with page number."""

    def test_single_page_native_text(self):
        page = _make_fake_page(0, "Hello native world")
        doc = _make_fake_doc([page])

        with patch("ingestion.extract.fitz") as mock_fitz:
            mock_fitz.open.return_value = doc
            mock_fitz.Matrix.return_value = MagicMock()
            result = Text_Extractor().extract("dummy.pdf")

        assert len(result) == 1
        assert result[0].page_number == 1
        assert result[0].text == "Hello native world"

    def test_multi_page_native_text_page_numbers(self):
        pages = [
            _make_fake_page(0, "Page one text"),
            _make_fake_page(1, "Page two text"),
            _make_fake_page(2, "Page three text"),
        ]
        doc = _make_fake_doc(pages)

        with patch("ingestion.extract.fitz") as mock_fitz:
            mock_fitz.open.return_value = doc
            result = Text_Extractor().extract("dummy.pdf")

        assert [r.page_number for r in result] == [1, 2, 3]
        assert result[1].text == "Page two text"

    def test_native_text_whitespace_only_treated_as_empty(self):
        """A page whose native text is only whitespace → routes to OCR (Req 1.2)."""
        page = _make_fake_page(0, "   \n\t  ")
        doc = _make_fake_doc([page])

        with patch("ingestion.extract.fitz") as mock_fitz, \
             patch("ingestion.extract._ocr_page_full", return_value="ocr result") as mock_ocr:
            mock_fitz.open.return_value = doc
            result = Text_Extractor().extract("dummy.pdf")

        mock_ocr.assert_called_once()
        assert result[0].text == "ocr result"


class TestTextExtractorFullPageOCR:
    """Requirement 1.2 – zero-char page goes to full-page OCR."""

    def test_zero_char_page_routed_to_ocr(self):
        page = _make_fake_page(0, "")
        doc = _make_fake_doc([page])

        with patch("ingestion.extract.fitz") as mock_fitz, \
             patch("ingestion.extract._ocr_page_full", return_value="scanned text") as mock_ocr:
            mock_fitz.open.return_value = doc
            result = Text_Extractor().extract("dummy.pdf")

        mock_ocr.assert_called_once_with(page)
        assert result[0].text == "scanned text"

    def test_native_text_page_does_not_call_full_page_ocr(self):
        """Pages with native text must NOT trigger full-page OCR."""
        page = _make_fake_page(0, "digital text here")
        doc = _make_fake_doc([page])

        with patch("ingestion.extract.fitz") as mock_fitz, \
             patch("ingestion.extract._ocr_page_full") as mock_full_ocr:
            mock_fitz.open.return_value = doc
            Text_Extractor().extract("dummy.pdf")

        mock_full_ocr.assert_not_called()

    def test_full_page_ocr_failure_yields_empty_text(self):
        """If full-page OCR raises, the page gets empty text and no crash."""
        page = _make_fake_page(0, "")
        doc = _make_fake_doc([page])

        with patch("ingestion.extract.fitz") as mock_fitz, \
             patch("ingestion.extract._ocr_page_full", side_effect=RuntimeError("tesseract gone")):
            mock_fitz.open.return_value = doc
            result = Text_Extractor().extract("dummy.pdf")

        assert result[0].text == ""

    def test_zero_char_page_does_not_call_embedded_image_ocr(self):
        """Full-page OCR path must not also try to OCR embedded images."""
        page = _make_fake_page(0, "")
        doc = _make_fake_doc([page])

        with patch("ingestion.extract.fitz") as mock_fitz, \
             patch("ingestion.extract._ocr_page_full", return_value="scanned"), \
             patch("ingestion.extract._ocr_embedded_images") as mock_img_ocr:
            mock_fitz.open.return_value = doc
            Text_Extractor().extract("dummy.pdf")

        mock_img_ocr.assert_not_called()


class TestTextExtractorEmbeddedImageOCR:
    """Requirements 1.3 & 1.4 – image OCR wrapped with Semantic Markdown Wrapping."""

    def test_image_ocr_appended_with_prefix(self):
        page = _make_fake_page(0, "Native text on page.")
        doc = _make_fake_doc([page])

        with patch("ingestion.extract.fitz") as mock_fitz, \
             patch("ingestion.extract._ocr_embedded_images",
                   return_value=[SEMANTIC_IMAGE_PREFIX + "diagram caption"]):
            mock_fitz.open.return_value = doc
            result = Text_Extractor().extract("dummy.pdf")

        text = result[0].text
        assert "Native text on page." in text
        assert SEMANTIC_IMAGE_PREFIX + "diagram caption" in text

    def test_image_snippets_appended_after_native_text(self):
        """Native text comes before the wrapped image snippet."""
        page = _make_fake_page(0, "Before")
        doc = _make_fake_doc([page])
        snippet = SEMANTIC_IMAGE_PREFIX + "after"

        with patch("ingestion.extract.fitz") as mock_fitz, \
             patch("ingestion.extract._ocr_embedded_images", return_value=[snippet]):
            mock_fitz.open.return_value = doc
            result = Text_Extractor().extract("dummy.pdf")

        text = result[0].text
        assert text.index("Before") < text.index(SEMANTIC_IMAGE_PREFIX)

    def test_no_images_no_appended_text(self):
        page = _make_fake_page(0, "Clean native page.")
        doc = _make_fake_doc([page])

        with patch("ingestion.extract.fitz") as mock_fitz, \
             patch("ingestion.extract._ocr_embedded_images", return_value=[]):
            mock_fitz.open.return_value = doc
            result = Text_Extractor().extract("dummy.pdf")

        assert result[0].text == "Clean native page."

    def test_multiple_image_snippets_all_appended(self):
        page = _make_fake_page(0, "Base text")
        doc = _make_fake_doc([page])
        snippets = [
            SEMANTIC_IMAGE_PREFIX + "img1",
            SEMANTIC_IMAGE_PREFIX + "img2",
        ]

        with patch("ingestion.extract.fitz") as mock_fitz, \
             patch("ingestion.extract._ocr_embedded_images", return_value=snippets):
            mock_fitz.open.return_value = doc
            result = Text_Extractor().extract("dummy.pdf")

        text = result[0].text
        assert "img1" in text
        assert "img2" in text

    def test_embedded_image_ocr_failure_does_not_crash(self):
        page = _make_fake_page(0, "Native only")
        doc = _make_fake_doc([page])

        with patch("ingestion.extract.fitz") as mock_fitz, \
             patch("ingestion.extract._ocr_embedded_images",
                   side_effect=RuntimeError("pixmap error")):
            mock_fitz.open.return_value = doc
            result = Text_Extractor().extract("dummy.pdf")

        # native text preserved, no crash
        assert result[0].text == "Native only"


class TestTextExtractorErrorHandling:
    """Requirements 1.7 – unreadable PDFs raise ExtractionError."""

    def test_fitz_open_failure_raises_extraction_error(self):
        with patch("ingestion.extract.fitz") as mock_fitz:
            mock_fitz.open.side_effect = Exception("file not found")
            with pytest.raises(ExtractionError, match="Cannot open PDF"):
                Text_Extractor().extract("nonexistent.pdf")

    def test_encrypted_pdf_raises_extraction_error(self):
        doc = _make_fake_doc([], is_encrypted=True)

        with patch("ingestion.extract.fitz") as mock_fitz:
            mock_fitz.open.return_value = doc
            with pytest.raises(ExtractionError, match="encrypted"):
                Text_Extractor().extract("locked.pdf")

        doc.close.assert_called_once()

    def test_doc_closed_after_extraction(self):
        page = _make_fake_page(0, "text")
        doc = _make_fake_doc([page])

        with patch("ingestion.extract.fitz") as mock_fitz:
            mock_fitz.open.return_value = doc
            Text_Extractor().extract("dummy.pdf")

        doc.close.assert_called_once()

    def test_doc_closed_even_if_page_processing_raises(self):
        """Ensure doc.close() is called even when extraction raises mid-way."""
        page = _make_fake_page(0, "")
        doc = _make_fake_doc([page])

        with patch("ingestion.extract.fitz") as mock_fitz, \
             patch("ingestion.extract._ocr_page_full", side_effect=RuntimeError("boom")):
            mock_fitz.open.return_value = doc
            # Should NOT raise – the individual page OCR failure is caught gracefully
            result = Text_Extractor().extract("dummy.pdf")

        doc.close.assert_called_once()
        assert result[0].text == ""


class TestTextExtractorPageLimit:
    """Requirement 1.5 – support 1–2000 page documents; truncate beyond MAX_PAGES."""

    def test_exactly_max_pages_processed(self):
        pages = [_make_fake_page(i, f"page {i}") for i in range(MAX_PAGES)]
        doc = _make_fake_doc(pages)

        with patch("ingestion.extract.fitz") as mock_fitz:
            mock_fitz.open.return_value = doc
            result = Text_Extractor().extract("big.pdf")

        assert len(result) == MAX_PAGES
        assert result[0].page_number == 1
        assert result[-1].page_number == MAX_PAGES

    def test_over_max_pages_truncated_with_warning(self, caplog):
        """Documents larger than MAX_PAGES are truncated; a warning is emitted."""
        import logging

        total = MAX_PAGES + 10
        pages = [_make_fake_page(i, f"page {i}") for i in range(total)]
        doc = _make_fake_doc(pages)

        with patch("ingestion.extract.fitz") as mock_fitz, \
             caplog.at_level(logging.WARNING, logger="ingestion.extract"):
            mock_fitz.open.return_value = doc
            result = Text_Extractor().extract("huge.pdf")

        assert len(result) == MAX_PAGES
        assert any("2000" in rec.message for rec in caplog.records)

    def test_single_page_document(self):
        page = _make_fake_page(0, "only page")
        doc = _make_fake_doc([page])

        with patch("ingestion.extract.fitz") as mock_fitz:
            mock_fitz.open.return_value = doc
            result = Text_Extractor().extract("single.pdf")

        assert len(result) == 1
        assert result[0].page_number == 1

    def test_empty_document_zero_pages(self):
        doc = _make_fake_doc([])

        with patch("ingestion.extract.fitz") as mock_fitz:
            mock_fitz.open.return_value = doc
            result = Text_Extractor().extract("empty.pdf")

        assert result == []


class TestTextExtractorOCRRouting:
    """Verify OCR-routing logic across a mixed document (Req 1.1, 1.2)."""

    def test_mixed_document_routing(self):
        """Native page, scanned page, native-with-image page – all handled correctly."""
        page_native = _make_fake_page(0, "Digital text")   # native
        page_scanned = _make_fake_page(1, "")              # zero chars → OCR
        page_with_img = _make_fake_page(2, "Native + img") # native + embedded image

        doc = _make_fake_doc([page_native, page_scanned, page_with_img])

        def fake_full_ocr(page):
            if page is page_scanned:
                return "ocr from scan"
            return ""

        def fake_img_ocr(page):
            if page is page_with_img:
                return [SEMANTIC_IMAGE_PREFIX + "chart text"]
            return []

        with patch("ingestion.extract.fitz") as mock_fitz, \
             patch("ingestion.extract._ocr_page_full", side_effect=fake_full_ocr), \
             patch("ingestion.extract._ocr_embedded_images",
                   side_effect=fake_img_ocr) as mock_img:
            mock_fitz.open.return_value = doc
            result = Text_Extractor().extract("mixed.pdf")

        assert result[0].text == "Digital text"
        assert result[1].text == "ocr from scan"
        assert SEMANTIC_IMAGE_PREFIX + "chart text" in result[2].text
        assert "Native + img" in result[2].text
        # _ocr_embedded_images should be called for native-text pages only
        # (native page AND native+image page), NOT for the scanned page
        assert mock_img.call_count == 2  # page_native and page_with_img
