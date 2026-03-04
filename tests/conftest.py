"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

import pytest

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))


# Sample vocabulary for testing
SAMPLE_VOCAB = [
    "the", "a", "an", "is", "are", "was", "were",
    "test", "example", "value", "number",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "10", "50", "100", "150", "200", "500", "1000",
    ".", ",", "-", "temperature", "pressure", "volume",
]


@pytest.fixture
def sample_vocab():
    """Provide a sample vocabulary for testing."""
    return SAMPLE_VOCAB


@pytest.fixture
def sample_texts():
    """Provide sample texts for testing."""
    return {
        "valid_numerical": "The value is 50",
        "invalid_high": "The value is 150",
        "invalid_low": "The value is -10",
        "multiple_numbers": "Values are 10, 20, and 30",
        "no_numbers": "This text has no numbers",
        "with_keywords": "Temperature and pressure are measured",
        "without_keywords": "Only general text here",
    }
