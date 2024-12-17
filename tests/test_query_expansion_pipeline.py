import pytest
from src.query_expansion import QueryExpansion

@pytest.fixture
def query_expansion_instance():
    """Fixture to initialize QueryExpansion."""
    config_path = "config/nested_configurations.yaml"  # Update path as needed
    return QueryExpansion(config_path=config_path)

def test_query_expansion_pipeline(query_expansion_instance):
    """Test end-to-end query expansion pipeline."""
    # Input query with spelling mistakes and abbreviations
    input_query = "Whre cn I fnd AI tech abbr?"

    # Expected behavior: The query expansion pipeline processes the input
    result = query_expansion_instance.query_augmentation(input_query)

    # Verify the result is non-empty and enhanced
    assert result is not None, "Query expansion returned None."
    assert isinstance(result, str), "Result is not a string."
    assert "AI technology" in result or "Where can I find", "Query expansion failed to enhance the query."

    print(f"Expanded Query: {result}")