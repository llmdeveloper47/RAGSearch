import sys
#sys.path.append('.')
import os
import yaml
import cohere
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()
# Access the API keys
COHERE_API_KEY = os.getenv("COHERE")


class CohereReranker:
    def __init__(self, config_path : str = '../config/nested_configurations.yaml'):
        self.ranking_model : str = None
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "../config/nested_configurations.yaml")
    
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self.config_path = config_path
        
        
    def get_config(self):
        with open(self.config_path, 'r') as yaml_file:
            configurations = yaml.safe_load(yaml_file)
            return configurations
        
    def initialize_reranker(self):

        if not COHERE_API_KEY:
            raise ValueError("API keys are not set. Check your .env file.")
        
        else:
            configurations = self.get_config()
            cohere_configuration = configurations['cohere']
            self.ranking_model = cohere_configuration['ranking_model']
            cohere_reranker = cohere.ClientV2(COHERE_API_KEY)
            
            return cohere_reranker
            
        
     
