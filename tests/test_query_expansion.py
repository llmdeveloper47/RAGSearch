import os
import pytest
from src.query_expansion import QueryExpansion

@pytest.fixture
def query_expansion_instance():
    # Initialize QueryExpansion instance
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "../config/nested_configurations.yaml")
    
    return QueryExpansion(config_path=config_path)

def test_correct_spelling(query_expansion_instance):
    """Test the `correct_spelling` method."""
    result = query_expansion_instance.correct_spelling(text_input="Ths is corectd txt")
    result = result.replace(".","")
    assert isinstance(result, str), "Result is not a string."
    assert result == "This is corrected text", "Spelling correction failed"
    assert len(result) > 0

def test_abbreviation_synonym_expansion(query_expansion_instance):
    """Test the `abbreviation_synonym_expansio` method."""
    result = query_expansion_instance.abbreviation_synonym_expansion(text_input="it's becuz that's blue")
    result = result.replace(".","")
    assert isinstance(result, str), "Result is not a string."
    assert result == "it is because that is blue"
    assert len(result) > 0

def test_topic_identification(query_expansion_instance):
    
    result = query_expansion_instance.topic_identification(text_input="black holes")
    assert isinstance(result, str), "Result is not a string."
    assert result == "black holes in astrophysics, their formation, properties, and effects on surrounding space and time"
    assert len(result) > 0
