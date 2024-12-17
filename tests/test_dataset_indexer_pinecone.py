import os
import yaml
import pytest
from src.dataset_indexer_pinecone import PineconeIndex

@pytest.fixture
def pinecone_index_instance():
    # Initialize QueryExpansion instance
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "../config/nested_configurations.yaml")
    
    with open(config_path, 'r') as yaml_file:
            configurations = yaml.safe_load(yaml_file)
    
    pinecone_config = configurations['pinecone']
    index_name = str(pinecone_config['index_name'])
    dimension = int(pinecone_config['dimension'])
    cloud = str(pinecone_config['cloud'])
    region = str(pinecone_config['region'])
    
    model_config = configurations["sentence_transformer"]
    embedding_model_name = model_config["model_name"]

    return PineconeIndex(config_path = config_path, index_name = index_name, embedding_model_name = embedding_model_name, dimension = dimension, cloud = cloud, region = region)


def test_correct_spelling(pinecone_index_instance):
      top_k = 3
      result = pinecone_index_instance.retrieve_documents(query_text="Who is the president of the US?" , top_k = top_k)
      documents = [item['chunk_text'] for item in result]                

      assert len(documents) == top_k

      top_k = 20
      result = pinecone_index_instance.retrieve_documents(query_text="Who is the president of the US?" , top_k = top_k)
      documents = [item['chunk_text'] for item in result]                

      assert len(documents) == top_k

