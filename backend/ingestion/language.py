"""
Language_Detector: ISO 639-1 language detection with confidence threshold.

Assigns a two-letter lowercase ISO 639-1 code when:
  - text length >= 20 characters, AND
  - detection confidence >= 0.50

Otherwise assigns the sentinel code "und" (undetermined).

Uses `langdetect` with a fixed DetectorFactory seed for reproducibility.

Requirements: 2.4, 2.5
"""

from langdetect import detect_langs, LangDetectException
from langdetect import DetectorFactory

# Seed for reproducibility (same text always yields the same result)
DetectorFactory.seed = 0

# Thresholds from requirements 2.4 / 2.5
_MIN_TEXT_LEN: int = 20
_MIN_CONFIDENCE: float = 0.50
_UNDETERMINED: str = "und"


class LanguageDetector:
    """Detects the primary language of a text string.

    Returns an ISO 639-1 two-letter lowercase language code (e.g. ``"en"``,
    ``"fr"``) when the text is long enough and the detection confidence meets
    the threshold; returns ``"und"`` otherwise.
    """

    def detect(self, text: str) -> str:
        """Return an ISO 639-1 code or ``"und"``.

        Parameters
        ----------
        text:
            The page or passage text to analyse.

        Returns
        -------
        str
            A two-letter lowercase ISO 639-1 code (e.g. ``"en"``) when
            ``len(text) >= 20`` and the top-ranked language probability is
            ``>= 0.50``; ``"und"`` in all other cases.
        """
        # Requirement 2.5: too short → und
        if len(text) < _MIN_TEXT_LEN:
            return _UNDETERMINED

        try:
            results = detect_langs(text)
        except LangDetectException:
            # Cannot determine language at all
            return _UNDETERMINED

        if not results:
            return _UNDETERMINED

        top = results[0]  # highest-probability language

        # Requirement 2.5: low confidence → und
        if top.prob < _MIN_CONFIDENCE:
            return _UNDETERMINED

        # Requirement 2.4: return as two-letter lowercase ISO 639-1 code
        return top.lang.lower()


# Module-level singleton for convenience
Language_Detector = LanguageDetector
