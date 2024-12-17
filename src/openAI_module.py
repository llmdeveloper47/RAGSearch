import sys
import os
import time
import yaml
import pickle
from tqdm import tqdm
from typing import List
import openai
from openai import OpenAI
from getpass import getpass
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()
# Access the API keys
OPENAI_API_KEY = os.getenv("OPENAI")

class OpenAIModule:
    """ This component initializes OpenAI GPT 4o across our application stack and is used in
        query expansion and genration in the RAG pipeline
    """
    
    def __init__(self, config_path : str = None):


        # Default path to config file if none provided
        if config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, "../config/nested_configurations.yaml")
    
        # Ensure the path is absolute
        self.config_path = os.path.abspath(config_path)

        # Validate if the config file exists
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

    
        self.embedding_model : str = None

        if not OPENAI_API_KEY:
            raise ValueError("API keys are not set. Check your .env file.")
        else:
            self.setup_openai_confguration()
            
    
    def setup_openai_confguration(self):
        """
        Prompt the user to input their OpenAI API key securely.
        """

        # Load the YAML file
        # 'nested_configurations.yaml'
        with open(self.config_path, 'r') as yaml_file:
            configurations = yaml.safe_load(yaml_file)

        # Access the OpenAI API key
        openai_config = configurations["openai"]
        #openai_api_key = #openai_config['api_key'] #load this from the .env file instead
        openai.api_key = OPENAI_API_KEY
        
        self.model = openai_config['model_name']
        self.embedding_model = openai_config['embedding_model']
        self.client = OpenAI(api_key = OPENAI_API_KEY)
                
    
    def get_embeddings_chunks(self, texts: List[str]) -> List[float]:
        
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(
                    input=text,
                    model=self.embedding_model
                )
            
            embeddings.append(response.data[0].embedding)
        return embeddings
    
    
    def get_embedding_query(self, text : str) -> List[float]:
        
        response = self.client.embeddings.create(
                    input=text,
                    model=self.embedding_model
                )
        return response.data[0].embedding
        
    
    def get_completion_gpt(self, prompt, model="gpt-4o-mini"):
        
        messages=[
                {"role": "system", "content": "You are a highly intelligent assistant."},
                {"role": "user", "content": prompt}
            ]

        #client = OpenAI(api_key = self.openai_api_key)
        

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0, # degree of randomness of the model's output
            #response_format={ "type": "json_object" }
        )
        return response.choices[0].message.content
    
    def get_response_gpt(self, prompt, model="gpt-4o-mini"):
        
        messages=[
                {"role": "system", "content": "You are a highly intelligent assistant."},
                {"role": "user", "content": prompt}
            ]

        #client = OpenAI(api_key = self.openai_api_key)
        

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0, # degree of randomness of the model's output
            response_format={ "type": "json_object" }
        )
        return response.choices[0].message.content
    
    
    
    
    
    
