import os
import pandas as pd
import yaml
import json
from tqdm import tqdm
from openAI_module import OpenAIModule
from dataset_builder import DatsetLoader

class QueryGenerator:
    """ This component builds on top of the DatsetLoader and OpenAIModule and samples 1000 rows from the 
        dataset, then uses GPT 4o to generate 5 question answer pairs for each document.
        The output is saved as a JSON file.
    """
    
    def __init__(self, config_path : str = '../config/nested_configurations.yaml'):
        self.openai_api_key : str = ""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(current_dir, "../config/nested_configurations.yaml")
    
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        self.open_ai_object :  OpenAIModule = OpenAIModule(config_path = self.config_path)

        with open(self.config_path, 'r') as yaml_file:
            configurations = yaml.safe_load(yaml_file)

        # Access the OpenAI API key
        
        dataset_config = configurations["huggingface"]
        dataset_name = dataset_config["dataset_name"]
        split = dataset_config["train_split"]
        file_name = dataset_config["file_name"]
        save_directory_path = dataset_config["save_directory_path"]
        dataset_size = int(dataset_config['dataset_size'])
        
        self.data_object = DatsetLoader(dataset_name  = dataset_name, save_directory_path = save_directory_path, file_name = file_name, dataset_size = dataset_size)
        


    def generate_queries(self, text_input : str) -> str:
        question_answer_pair_prompt = f"""
                    You are an expert english language assistant. Your task is to understand the user provided document
                    and generate 5 questions and their respective answers from the document. Please take your time
                    in understand the document and then generate the question answer pairs. Please remember that 
                    for a given question, the respective answer can be from different parts of the document.

                    DO NOT DEVIATE FROM THE ABOVE ASSUMPTIONS.
                    
                    Please output the result as a JSON file with the questions as keys and respective answers as values.
                    
                    User_Document:
                    ```{text_input}```
                    """
        response = self.open_ai_object.get_response_gpt(prompt = question_answer_pair_prompt)
        return response
    
    def select_documents(self) -> pd.DataFrame:

        temp = self.data_object.dataset
        temp['length'] = temp['text'].apply(lambda x : len(x))
        temp_subset = temp[temp['length'] <= 128000] # 128000 -> max context window size of GPT
        temp_subset = temp_subset.sample(n=100, random_state = 412) # we sample 1000 rows from the dataframe
        temp_subset = temp_subset.reset_index().iloc[:,1:]
        return temp_subset
    
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of search_cli.py
    config_path = os.path.join(current_dir, "../config/nested_configurations.yaml")
    
    query_generator = QueryGenerator(config_path = config_path)
    data = query_generator.select_documents()
    response_list = []
    for i in tqdm(range(len(data))):
        text_input = data.iloc[i,1]
        response = query_generator.generate_queries(text_input)
        result = json.loads(response)
        response_list.append(result)
        
    final_result = {k: v for d in response_list for k, v in d.items()}    
    if os.path.isdir(f"../evaluation"):
        with open("../evaluation/sample_query_answer_pairs.json", "w") as outfile: 
            json.dump(final_result, outfile)
    else:
        os.mkdir(f"../evaluation")
        with open("../evaluation/sample_query_answer_pairs.json", "w") as outfile: 
            json.dump(final_result, outfile)


    
