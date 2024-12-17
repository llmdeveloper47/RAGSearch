import os
import pytest
from src.cohere_reranker_module import CohereReranker


@pytest.fixture
def cohere_reranker_instance():
    # Initialize QueryExpansion instance
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "../config/nested_configurations.yaml")
    
    return CohereReranker(config_path=config_path)

def test_rank_documents_with_cohere(cohere_reranker_instance):
    """Test the `cohere reranker` method."""


    augmented_query = 'What is the capital of the United States?'
    documents = ['The capital of the United States is Washington, D.C., which stands for the District of Columbia.',
                 'Washington, D.C. serves as the political center of the U.S., housing key institutions like the White House, the Capitol Building, and the Supreme Court.',
                 'Founded in 1790, Washington, D.C. was established as a planned city to serve as the nation’s capital, separate from any state.',
                 'The city is home to iconic landmarks, including the Washington Monument, the Lincoln Memorial, and numerous Smithsonian museums.',
                 'As the capital, Washington, D.C. hosts foreign embassies, federal government agencies, and major national events like presidential inaugurations.']

    reranker = cohere_reranker_instance.initialize_reranker()
    response = reranker.rerank(model = cohere_reranker_instance.ranking_model, query = augmented_query, documents= documents, top_n=len(documents))
    ranks = []
    for i in range(len(response.results)):
        ranks.append(response.results[i].index)
    
    assert len(ranks) == len(documents)
    
