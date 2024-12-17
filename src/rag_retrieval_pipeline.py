import os
import sys
import yaml
from typing import List
from openAI_module import OpenAIModule
from query_expansion import QueryExpansion
from cohere_reranker_module import CohereReranker
from dataset_indexer_pinecone import PineconeIndex

import warnings
from transformers import logging
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

class RAGPipeline:
    """ This component uses the OpenAIModule, QueryExpansion, CohereReranker and PineconeIndex components to intialize OpenAI GPT 4o, cohere's reranker, connects to 
        the Pinecone index and enhances the user provided query to generate a concise answer for the query
    """
    
    def __init__(self, config_path : str = '../config/nested_configurations.yaml'):

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(current_dir, "../config/nested_configurations.yaml")
    
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        #self.config_path = config_path
        self.top_k = 10
        self.indexer_object : PineconeIndex = None
        self.initialize_vector_database()
        self.query_expansion_object : QueryExpansion = QueryExpansion(config_path = self.config_path)
        self.open_ai_object : OpenAIModule = OpenAIModule(config_path = self.config_path)
        self.cohere_object : CohereReranker = CohereReranker(config_path = self.config_path)

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
        model_path = model_config["model_path"]
        model_path = f'./models/{model_path}'
        
        self.indexer_object = PineconeIndex(config_path = self.config_path, index_name = index_name, embedding_model_name = embedding_model_name, dimension = dimension, cloud = cloud, region = region)

    def construct_rag_prompt(self,documents, query, additional_input="") -> str:
        """
        Constructs a detailed prompt for GPT-4 to act as a RAG system.

        Args:
            documents (list): List of retrieved documents or passages.
            query (str): User's query.
            additional_input (str): Additional user-provided input.

        Returns:
            str: The constructed prompt.
        """
        # Header and instructions
        prompt_header = """
    You are a highly intelligent and context-aware assistant designed to provide detailed and accurate answers based on retrieved documents. Your task is to:

    1. Carefully read and analyze the provided **context documents**.
    2. Extract and synthesize relevant information from the documents that is most pertinent to the user's query.
    3. If the context documents do not provide enough information, please use your own general knowledge and assumptions to answer the query. 
    4. Generate a comprehensive, concise, and accurate answer tailored to the query, supported by the information in the provided documents whenever possible.

    ---
    """
        # Input structure
        context_intro = "### **Input Structure**:\n"
        context_intro += "**Context Documents**:\n"
        for i, doc in enumerate(documents, start=1):
            context_intro += f"{i}. {doc}\n"

        context_intro += f"\n**User Query**:\n{query}\n"
        if additional_input:
            context_intro += f"\n**Input Text Sequence**:\n{additional_input}\n"

        # Guidelines for the output
        output_structure = """
    ---
    
     Using the documents provided and your own general knowledge, please provide a direct and concise answer to the query. 
     Please note that if the documents do not fully answer the query, then provide an answer based on your own general knowledge. 
     
     Please do not deviate from the above constraints.
     
     Only provide the final output.
    """

        # Combine all parts
        prompt = prompt_header + context_intro + output_structure
        return prompt    
    
    def rank_documents_with_cohere(self, query : str, documents : List[str]) -> List[str]:
        """
        Use Cohere's re-ranker to rank a list of documents based on relevance to a query.

        Args:
            api_key (str): Your Cohere API key.
            query (str): The user query to rank documents against.
            documents (list): List of documents to be ranked.

        Returns:
            list: Ranked documents with their scores.
        """
        # Initialize Cohere client
        reranker = self.cohere_object.initialize_reranker()

        # Call the re-rank API
        response = reranker.rerank(
            model=self.cohere_object.ranking_model,
            query=query,
            documents=documents,
            top_n=len(documents)  # Return rankings for all documents
        )
        
        ranks = []
        for i in range(len(response.results)):
            ranks.append(response.results[i].index)


        def sort_by_indexes(lst, indexes):
              return [val for (_, val) in sorted(zip(indexes, lst), key=lambda x: x[0])]
            
        sorted_documents = sorted_lst = sort_by_indexes(documents, ranks)
        
        return sorted_documents

        
    def generate_answer_with_rag(self, text_input : str) -> str:
        
        # augment the user provided query
        
        augmented_query = self.query_expansion_object.query_augmentation(text_input = text_input)
        
        result = self.indexer_object.retrieve_documents(query_text = augmented_query, top_k = self.top_k)
        documents = [item['chunk_text'] for item in result]                
        
        sorted_documents = self.rank_documents_with_cohere(query = augmented_query, documents = documents)
        
        additional_input = "Answer the question as accurately as possible."
                
        rag_prompt = self.construct_rag_prompt(documents = sorted_documents, query = augmented_query, additional_input=additional_input)
        
        response = self.open_ai_object.get_completion_gpt(prompt = rag_prompt)
        return response
    
