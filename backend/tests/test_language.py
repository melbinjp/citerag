"""
Unit tests for LanguageDetector (backend/ingestion/language.py).

These tests cover:
  - Text shorter than 20 chars → "und"  (Req 2.5)
  - Text exactly 20 chars long with detectable language
  - Confident detection of known languages  (Req 2.4)
  - Undetectable / noisy text → "und"  (Req 2.5)
  - Return value is always a two-letter lowercase string or "und"  (Req 2.4)
"""

import pytest

from backend.ingestion.language import LanguageDetector

detector = LanguageDetector()


# ---------------------------------------------------------------------------
# Short-text → "und"
# ---------------------------------------------------------------------------

class TestShortText:
    def test_empty_string_returns_und(self):
        assert detector.detect("") == "und"

    def test_single_word_returns_und(self):
        assert detector.detect("Hello") == "und"

    def test_nineteen_chars_returns_und(self):
        # 19 characters — just below threshold
        assert detector.detect("This is 19 char tx!") == "und"

    def test_exactly_20_chars_eligible(self):
        # 20 characters — meets length threshold; should not return "und"
        # (actual language depends on content, but must not be length-blocked)
        text = "This is twenty chars"  # exactly 20
        assert len(text) == 20
        result = detector.detect(text)
        # With sufficient confidence this English text should be "en"
        assert result == "en"


# ---------------------------------------------------------------------------
# Confident language detection
# ---------------------------------------------------------------------------

class TestLanguageDetection:
    def test_english_detected(self):
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a well-known English pangram."
        )
        assert detector.detect(text) == "en"

    def test_spanish_detected(self):
        text = (
            "El rápido zorro marrón salta sobre el perro perezoso. "
            "Esta es una oración en español."
        )
        assert detector.detect(text) == "es"

    def test_french_detected(self):
        text = (
            "Le renard brun rapide saute par-dessus le chien paresseux. "
            "Ceci est une phrase en français."
        )
        assert detector.detect(text) == "fr"

    def test_german_detected(self):
        text = (
            "Der schnelle braune Fuchs springt über den faulen Hund. "
            "Dies ist ein Satz auf Deutsch."
        )
        assert detector.detect(text) == "de"

    def test_return_value_is_two_letter_lowercase(self):
        text = "This is a sufficiently long English sentence for detection."
        result = detector.detect(text)
        assert result != "und"
        assert len(result) == 2
        assert result == result.lower()


# ---------------------------------------------------------------------------
# Noisy / undetectable text → "und"
# ---------------------------------------------------------------------------

class TestUndetermined:
    def test_random_symbols_returns_und(self):
        # A string >= 20 chars but composed of symbols with no clear language
        text = "!@#$%^&*()_+{}|:<>?!@#$"
        result = detector.detect(text)
        # Either library raises or assigns low confidence → "und"
        assert result == "und"

    def test_mixed_script_noise_long(self):
        # Mix of very different scripts unlikely to yield >= 0.50 confidence
        # for any single language; at minimum must return a valid value
        text = "αβγδ 12345 あいうえお ABCDE xyz!!"
        result = detector.detect(text)
        assert isinstance(result, str)
        assert result == "und" or (len(result) == 2 and result == result.lower())


# ---------------------------------------------------------------------------
# Idempotency / reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_text_same_result(self):
        text = "Reproducibility is important for language detection in pipelines."
        result1 = detector.detect(text)
        result2 = detector.detect(text)
        assert result1 == result2
