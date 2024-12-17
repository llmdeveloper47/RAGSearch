import os
import sys
#sys.path.append('.')
import yaml
import json
from tqdm import tqdm
from dataset_builder import DatsetLoader
from dataset_indexer_pinecone import PineconeIndex
from typing import List
from transformers import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")

class WikipediaIndex:
    """This class uses the DatsetLoader and PineconeIndex class to load a dataset from huggingface ( or from disk )
    and then create an index from the dataset. In case the index ever gets deleted ( which it won't as I have enabled deletion safe config, This needs to run before we start querying our system"
    """
    
    def __init__(self, config_path : str = '../config/nested_configurations.yaml'):

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(current_dir, "../config/nested_configurations.yaml")
    
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")


        #self.config_path = config_path
        self.indexer_object : PineconeIndex = None
        self.data_object : DatsetLoader = None
        
        self.initialize_vector_database()
        self.initialize_dataset_loader()
    
    def get_config(self):
        with open(self.config_path, 'r') as yaml_file:
            configurations = yaml.safe_load(yaml_file)
            
        return configurations
    
    def initialize_vector_database(self) -> None:
        configurations = self.get_config()
        
        pinecone_config = configurations['pinecone']
        index_name = str(pinecone_config['index_name'])
        dimension = int(pinecone_config['dimension'])
        cloud = str(pinecone_config['cloud'])
        region = str(pinecone_config['region'])
        
        model_config = configurations["sentence_transformer"]
        embedding_model_name = model_config["model_name"]
        
        self.indexer_object = PineconeIndex(config_path = self.config_path, index_name = index_name, embedding_model_name = embedding_model_name, dimension = dimension, cloud = cloud, region = region)
    
    
    def initialize_dataset_loader(self) -> None:
        
        configurations = self.get_config()
        dataset_config = configurations["huggingface"]
        dataset_name = dataset_config["dataset_name"]
        split = dataset_config["train_split"]
        file_name = dataset_config["file_name"]
        save_directory_path = dataset_config["save_directory_path"]
        dataset_size = int(dataset_config['dataset_size'])
        
        self.data_object = DatsetLoader(dataset_name  = dataset_name, save_directory_path = save_directory_path, file_name = file_name, dataset_size = dataset_size)
        
    def upsert_dataset(self) -> None:
            
        index = 0
        for row in tqdm(self.data_object.dataset.itertuples(index=False), total=len(self.data_object.dataset)):
            # Access row fields by attribute name or index
            document_id = f'document_id_{index}'
            title = row.title
            text = row.text
            self.indexer_object.upsert_document(document_id = document_id, title = title, text = text)
            index = index + 1
            
        print('All Records Added To Index')
        
if __name__ == "__main__":
    
    # load or create the index if it doesn't exists
    print('loading/creating pinecone index')
    index_builder = WikipediaIndex(config_path = 'nested_configurations.yaml')
    
    # once the index is created we start upserting the dataset into the index
    print('adding documents to index')
    index_builder.upsert_dataset()
  