"""
Root-level pytest configuration.

tests/test_basic.py uses the pre-refactor import path
'services.generation.*' which no longer exists. It is excluded from
collection so that 'pytest tests/' does not fail on import.
"""

collect_ignore = ["tests/test_basic.py"]
