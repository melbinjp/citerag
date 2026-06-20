"""PDF text extraction and targeted OCR for the RAG PDF ingestion pipeline.

Implements:
- Requirement 1.1: PyMuPDF native per-page text extraction with page-number association.
- Requirement 1.2: Route a page to full-page OCR when native extraction yields zero chars.
- Requirement 1.3: OCR embedded raster image bounding boxes and append to the page text.
- Requirement 1.4: Wrap OCR'd image text with the Semantic Markdown Wrapping system-note
  marker ``> [System Note: Image Content] ...``.
- Requirement 1.5: Support PDF documents containing 1 to 2000 pages.

The module intentionally avoids importing ``pytesseract`` at module level so that the
module can be imported in environments where Tesseract is not installed (the caller is
responsible for handling the resulting errors at runtime if OCR is attempted in such
environments).
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import fitz  # PyMuPDF – required dependency (pymupdf>=1.23.0 in requirements.txt)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────

#: Semantic Markdown Wrapping prefix applied to every OCR-recovered image passage.
#: (Requirements 1.3, 1.4)
SEMANTIC_IMAGE_PREFIX = "> [System Note: Image Content] "

#: Maximum supported page count (Requirement 1.5).
MAX_PAGES = 2000


@dataclass(frozen=True)
class PageText:
    """Text extracted (or OCR'd) from a single PDF page.

    Attributes:
        page_number: 1-based page number within the source PDF.
        text: Extracted text for the page.  May contain Semantic Markdown
            Wrapping lines appended after native text for image-derived content.
    """

    page_number: int
    text: str


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _ocr_pixmap(pixmap) -> str:  # noqa: ANN001
    """Run Tesseract OCR on a PyMuPDF ``Pixmap`` and return the resulting string.

    Args:
        pixmap: A ``fitz.Pixmap`` instance to OCR.

    Returns:
        The OCR'd text, stripped of leading/trailing whitespace.  Returns an
        empty string if Tesseract returns no output.

    Raises:
        ImportError: If ``pytesseract`` or ``Pillow`` are not installed.
        pytesseract.TesseractNotFoundError: If the Tesseract binary is missing.
    """
    import pytesseract  # deferred import – not required at module load time
    from PIL import Image  # noqa: PLC0415

    # Convert the pixmap to a PIL Image without writing to disk.
    pil_image = Image.open(io.BytesIO(pixmap.tobytes("png")))
    result = pytesseract.image_to_string(pil_image)
    return result.strip() if result else ""


def _ocr_page_full(page) -> str:  # noqa: ANN001
    """Render the entire *page* at 2× resolution and OCR the resulting image.

    The 2× (144 DPI) render improves Tesseract accuracy compared with the
    default 72 DPI render without becoming prohibitively large.

    Args:
        page: A ``fitz.Page`` instance.

    Returns:
        OCR'd text stripped of surrounding whitespace.
    """
    # matrix=fitz.Matrix(2, 2) doubles the resolution (72 → 144 DPI).
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    return _ocr_pixmap(pixmap)


def _ocr_embedded_images(page) -> list[str]:  # noqa: ANN001
    """OCR every embedded raster image on *page* and return wrapped text snippets.

    For each embedded image whose bounding box can be resolved on the page, the
    image pixels are extracted via PyMuPDF, passed to Tesseract, and — if the
    result is non-empty — wrapped with :data:`SEMANTIC_IMAGE_PREFIX` per
    Requirements 1.3 and 1.4.

    Args:
        page: A ``fitz.Page`` instance.

    Returns:
        List of Semantic Markdown Wrapped strings (one per image that yielded
        non-empty OCR output).  Empty list if the page has no embedded images or
        all of them yield empty OCR output.
    """
    doc = page.parent  # the fitz.Document that owns this page
    wrapped_snippets: list[str] = []

    # get_images() returns a list of (xref, smask, width, height, bpc, colorspace,
    # alt_colorspace, name, filter, referencer) tuples.
    image_list = page.get_images(full=True)
    if not image_list:
        return wrapped_snippets

    for img_info in image_list:
        xref: int = img_info[0]

        try:
            # Retrieve the image rect(s) on the page.  get_image_rects() returns a
            # list of fitz.Rect objects representing where the image is placed.
            rects = page.get_image_rects(xref)
            if not rects:
                logger.debug(
                    "Page %d: image xref=%d has no rects on this page; skipping.",
                    page.number + 1,
                    xref,
                )
                continue

            # Use the first (and typically only) placement rect.
            rect = rects[0]
            if rect.is_empty or rect.is_infinite:
                continue

            # Extract the image from the document by its xref.
            image_dict = doc.extract_image(xref)
            if not image_dict:
                continue

            img_bytes: bytes = image_dict.get("image", b"")
            if not img_bytes:
                continue

            # Build a pixmap from the raw image bytes.
            pixmap = fitz.Pixmap(img_bytes)
            # Tesseract requires RGB/RGBA; convert if the pixmap is CMYK or
            # has an alpha channel mixed in an unsupported colour space.
            if pixmap.n > 4 or pixmap.colorspace is None:
                pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
            elif pixmap.n == 4:
                # May be RGBA – ensure it is in a Tesseract-friendly format.
                pixmap = fitz.Pixmap(fitz.csRGB, pixmap)

            ocr_text = _ocr_pixmap(pixmap)
            if ocr_text:
                wrapped = SEMANTIC_IMAGE_PREFIX + ocr_text
                wrapped_snippets.append(wrapped)

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Page %d: failed to OCR image xref=%d: %s",
                page.number + 1,
                xref,
                exc,
            )

    return wrapped_snippets


# ──────────────────────────────────────────────────────────────────────────────
# Public interface
# ──────────────────────────────────────────────────────────────────────────────


class ExtractionError(Exception):
    """Raised when a PDF cannot be opened or is otherwise unprocessable.

    Callers (e.g. the ingestion pipeline) should catch this exception, record an
    ingestion error entry, and continue with the next document.
    """


class Text_Extractor:
    """Extract text from PDF documents using PyMuPDF with targeted Tesseract OCR.

    Extraction strategy per page (Requirements 1.1–1.4):

    1. **Native text** – ``page.get_text()`` via PyMuPDF.  The result is
       associated with the 1-based page number (Requirement 1.1).
    2. **Full-page OCR** – if the native text is empty (zero characters after
       stripping whitespace), the page is rendered and passed to Tesseract
       (Requirement 1.2).
    3. **Embedded-image OCR** – if the page *does* have native text, each
       embedded raster image is separately OCR'd and the result, wrapped with
       the Semantic Markdown Wrapping prefix ``> [System Note: Image Content] ``,
       is appended to the native text (Requirements 1.3, 1.4).

    Page-count limit: documents with more than :data:`MAX_PAGES` (2 000) pages
    are supported; pages beyond that limit are silently skipped with a warning
    (Requirement 1.5 specifies *support* for 1–2 000 pages, not rejection of
    larger documents).

    Usage::

        extractor = Text_Extractor()
        pages = extractor.extract("path/to/document.pdf")
        for page in pages:
            print(page.page_number, page.text[:80])

    Raises:
        ExtractionError: When the PDF cannot be opened (corrupt, encrypted, or
            not a valid PDF file).
    """

    def extract(self, pdf_path: str) -> list[PageText]:
        """Extract text from every page of *pdf_path*.

        Args:
            pdf_path: Absolute or relative filesystem path to a PDF file.

        Returns:
            List of :class:`PageText` instances, one per page processed, in
            page order.  Pages that yield no text at all (neither native nor
            OCR) are still included with an empty ``text`` field so that the
            caller can detect and log them.

        Raises:
            ExtractionError: If the PDF cannot be opened.
        """
        # ── Open the PDF ─────────────────────────────────────────────────────
        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:  # noqa: BLE001
            raise ExtractionError(
                f"Cannot open PDF '{pdf_path}': {exc}"
            ) from exc

        # Reject encrypted PDFs that PyMuPDF opened but cannot decrypt.
        if doc.is_encrypted:
            doc.close()
            raise ExtractionError(
                f"Cannot open PDF '{pdf_path}': document is encrypted/password-protected."
            )

        results: list[PageText] = []

        try:
            total_pages = len(doc)

            # Requirement 1.5: support 1–2000 page documents; warn if more pages exist.
            pages_to_process = total_pages
            if total_pages > MAX_PAGES:
                logger.warning(
                    "'%s' has %d pages; only the first %d will be processed "
                    "(Requirement 1.5).",
                    pdf_path,
                    total_pages,
                    MAX_PAGES,
                )
                pages_to_process = MAX_PAGES

            for page_index in range(pages_to_process):
                page = doc[page_index]
                page_number = page_index + 1  # 1-based

                # ── Step 1: native text extraction (Requirement 1.1) ─────────
                native_text: str = page.get_text()  # returns "" if no text layer

                if not native_text.strip():
                    # ── Step 2: full-page OCR (Requirement 1.2) ──────────────
                    logger.debug(
                        "Page %d of '%s': zero native chars → full-page OCR.",
                        page_number,
                        pdf_path,
                    )
                    try:
                        ocr_text = _ocr_page_full(page)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Page %d of '%s': full-page OCR failed: %s",
                            page_number,
                            pdf_path,
                            exc,
                        )
                        ocr_text = ""

                    page_text = ocr_text

                else:
                    # ── Step 3: OCR embedded images (Requirements 1.3, 1.4) ──
                    try:
                        image_snippets = _ocr_embedded_images(page)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Page %d of '%s': embedded-image OCR failed: %s",
                            page_number,
                            pdf_path,
                            exc,
                        )
                        image_snippets = []

                    if image_snippets:
                        # Append each wrapped image snippet on its own line
                        # after the native text.
                        combined_parts = [native_text] + image_snippets
                        page_text = "\n".join(combined_parts)
                    else:
                        page_text = native_text

                results.append(PageText(page_number=page_number, text=page_text))

        finally:
            doc.close()

        return results
