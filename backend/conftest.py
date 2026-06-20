"""conftest.py — pytest path configuration for the backend test suite.

Adds two paths to sys.path so that all test imports resolve correctly:

  1. The workspace root (ackathon/) — allows ``from backend.ingestion.xxx``
     style imports used by test_clean.py, test_language.py, and
     tests/ingestion/test_error_log.py.

  2. The backend/ directory itself — allows ``from ingestion.xxx`` and
     ``from data_models import ...`` style imports used by
     test_chunker_config.py and test_chunker_chunk.py.
"""

import sys
import os

# backend/ directory (this file's own directory)
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# workspace root = parent of backend/
WORKSPACE_ROOT = os.path.dirname(BACKEND_DIR)

for path in (BACKEND_DIR, WORKSPACE_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)
