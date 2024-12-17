import os
import pytest
from src.openAI_module import OpenAIModule

@pytest.fixture
def openai_instance():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "../config/nested_configurations.yaml")
    
    return OpenAIModule(config_path=config_path)

def test_rank_documents_with_cohere(openai_instance):

    prompt = 'Please generate 1 line on UFOs'
    response = openai_instance.get_completion_gpt(prompt = prompt)
    assert len(response) > 0

