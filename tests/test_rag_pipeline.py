import pytest
from src.rag_retrieval_pipeline import RAGPipeline

@pytest.fixture
def rag_pipeline_instance():
    """Fixture to initialize RAG pipeline."""
    config_path = "config/nested_configurations.yaml"  # Update path as needed
    return RAGPipeline(config_path=config_path)

def test_rag_retrieval_pipeline(rag_pipeline_instance):
    """Test end-to-end RAG retrieval pipeline."""
    # Input user query
    user_query = "Latest trends in AI technology?"

    # Execute the RAG pipeline
    response = rag_pipeline_instance.generate_answer_with_rag(user_query)

    # Validate response
    assert response is not None, "RAG pipeline returned None."
    assert isinstance(response, str), "Response is not a string."
    assert len(response) > 0, "No documents retrieved or reranked."

    