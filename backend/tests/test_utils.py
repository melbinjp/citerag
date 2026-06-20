import pytest
from utils.splitter import split_text
from utils.loaders import load_source
from utils.exceptions import DocumentLoaderError

# --- Tests for utils.splitter ---

def test_split_text_standard():
    """Tests basic splitting with overlap by checking properties."""
    text = "This is the first sentence. This is the second sentence. This is the third."
    max_chars=30
    chunks = split_text(text, max_chars=max_chars, overlap=10)
    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk) <= max_chars

def test_split_text_short():
    """Tests text shorter than max_chars, which should not be split."""
    text = "This is a short text."
    chunks = split_text(text, max_chars=100, overlap=20)
    assert len(chunks) == 1
    assert chunks[0] == text

def test_split_text_empty():
    """Tests that an empty string results in an empty list."""
    text = ""
    chunks = split_text(text, max_chars=100, overlap=20)
    assert len(chunks) == 0

def test_split_text_whitespace():
    """Tests that leading/trailing/multiple whitespaces are handled."""
    text = "  leading and trailing whitespace  and  multiple   spaces  "
    max_chars=20
    chunks = split_text(text, max_chars=max_chars, overlap=5)

    # The splitter should have removed all excess whitespace internally
    cleaned_text = " ".join(text.split())
    assert chunks[0][0] != " " # First chunk should not start with space
    assert cleaned_text.startswith(chunks[0])
    for chunk in chunks:
        assert len(chunk) <= max_chars

# --- Tests for utils.loaders ---

def test_load_source_txt():
    """Tests loading a simple .txt file."""
    content = b"This is a test."
    text = load_source(content, ".txt")
    assert text == "This is a test."

def test_load_source_html():
    """Tests loading an HTML file and stripping scripts."""
    html_content = b"<html><head><script>alert('bad');</script></head><body><p>Hello</p></body></html>"
    text = load_source(html_content, ".html")
    assert "Hello" in text
    assert "alert" not in text

def test_load_source_unsupported_binary():
    """Tests that an unsupported binary file type raises an error."""
    # Using some random bytes that are not valid UTF-8 and cannot be converted
    binary_content = b'\x80\x81\x82'
    with pytest.raises(DocumentLoaderError):
        load_source(binary_content, ".bin")

def test_load_source_unsupported_but_decodable():
    """Tests a fake extension that is still valid text."""
    content = b"This is decodable."
    text = load_source(content, ".fake")
    assert text == "This is decodable."

def test_load_source_pdf_mocked(mocker):
    """Tests PDF loading by mocking PyMuPDF (fitz), which is the current fallback."""
    mock_fitz_open = mocker.patch('fitz.open')
    mock_doc = mocker.MagicMock()
    mock_page_a = mocker.MagicMock()
    mock_page_a.get_text.return_value = "This is a PDF page."
    mock_page_b = mocker.MagicMock()
    mock_page_b.get_text.return_value = "This is a PDF page."
    mock_doc.__iter__ = mocker.MagicMock(return_value=iter([mock_page_a, mock_page_b]))
    mock_fitz_open.return_value = mock_doc

    # Make MarkItDown return empty so the PyMuPDF fallback is triggered
    mock_md_instance = mocker.MagicMock()
    mock_md_result = mocker.MagicMock()
    mock_md_result.text_content = ""
    mock_md_instance.convert.return_value = mock_md_result
    mocker.patch('utils.loaders.MarkItDown', return_value=mock_md_instance)

    pdf_content = b'%PDF-1.4...'  # Dummy content
    text = load_source(pdf_content, ".pdf")

    assert "This is a PDF page." in text
    mock_fitz_open.assert_called_once()

def test_load_source_docx_mocked(mocker):
    """Tests DOCX loading by mocking MarkItDown, which is the current implementation."""
    mock_md_instance = mocker.MagicMock()
    mock_md_result = mocker.MagicMock()
    mock_md_result.text_content = "This is a DOCX paragraph.\nThis is a DOCX paragraph."
    mock_md_instance.convert.return_value = mock_md_result
    mocker.patch('utils.loaders.MarkItDown', return_value=mock_md_instance)

    docx_content = b'PK...'  # Dummy content
    text = load_source(docx_content, ".docx")

    assert text == "This is a DOCX paragraph.\nThis is a DOCX paragraph."
    mock_md_instance.convert.assert_called_once()
