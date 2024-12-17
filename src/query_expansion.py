import os
import sys
#sys.path.append('.')
import time
import yaml
import pickle
from tqdm import tqdm
import openai
from openai import OpenAI
from getpass import getpass
from openAI_module import OpenAIModule

class QueryExpansion:
    """ This component takes in the user query and applies the following steps to it to enhance it's meaning for improved retrieval:
        Step 1 - correct_spelling: Spelling Correction , incase the user mistakenly types an incorrect spelling of a word, the correct_spelling function corrects all
            spelling mistakes
        Step 2 - abbreviation_synonym_expansion : replaces all abbreviations with their full extended form to improve retrieval quality
        Step 3 - topic_identification : This component identifies the topic the output of Step 2 falls under and then returns the enhanced query
            
    """
    
    def __init__(self, config_path : str = '../config/nested_configurations.yaml'):
        self.openai_api_key : str = ""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(current_dir, "../config/nested_configurations.yaml")
    
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        self.open_ai_object :  OpenAIModule = OpenAIModule(config_path = self.config_path)
    
    def correct_spelling(self, text_input : str) -> str:
        spell_correction_prompt = f"""
                    You are a spell correction assistant. Correct spelling mistakes in user input 
                    while retaining its original meaning. Return only the corrected version.
                    User_Query:
                    ```{text_input}```
                    """
        response = self.open_ai_object.get_completion_gpt(prompt = spell_correction_prompt)
        return response
    
    def abbreviation_synonym_expansion(self, text_input : str) -> str:
        abbreviation_prompt = f"""
                    You are a english language assistant. Your task is to expand the user provided text input 
                    by replacing any abbreviations in the provided text with their complete and full form 
                    while retaining the meaning of the original text. 
                    Return only the corrected version and no special characters.
                    
                    PLEASE DO NOT DEVIATE FROM THE ABOVE ASSUMPTIONS. PLEASE MAKE SURE TO RETAIN THE ORIGINAL MEANING OF THE USER 
                    PROVIDED TEXT INPUT
                    User_Query:
                    ```{text_input}```
                    """
        response = self.open_ai_object.get_completion_gpt(prompt = abbreviation_prompt)
        return response
    
    
    def topic_identification(self, text_input : str) -> str:
        topic_identification_prompt = f"""
                    You are a highly intelligent assistant that specializes in analyzing and categorizing user provided queries. Your task is to:

                    Identify the primary topic of the user's query. Be specific and expand the original provided query with the identified main
                    topic for for enhanced semantic search. 
                    
                    PLEASE DO NOT DEVIATE FROM THE ABOVE ASSUMPTIONS. PLEASE REMEBER THAT OUR TASK IS QUERY EXPANSION FOR 
                    ENHANCED SEMANTIC SEARCH. PLEASE MAKE SURE TO RETAIN THE ORIGINAL MEANING OF THE USER PROVIDED TEXT INPUT AND OUTPUT
                    ONLY THE ENHANCED QUERY, NO OTHER EXTRA STRINGS OR CHARACTERS.
                    
                    User_Query:
                    ```{text_input}```
                    """
        response = self.open_ai_object.get_completion_gpt(prompt = topic_identification_prompt)
        return response
        
    
    def query_augmentation(self, text_input : str) -> str:
        corrected_query = self.correct_spelling(text_input = text_input)
        corrected_query = self.abbreviation_synonym_expansion(text_input = corrected_query)
        result = self.topic_identification(text_input = corrected_query)
        
        return result
    