import os
from datasets import load_dataset
from datasets import Dataset
import datasets
import pandas as pd
from tqdm import tqdm
from typing import List


class DatsetLoader:
    """Class to stream wikiedia dataset from huggingface hub and save the first 10k records
       from the dataset on disk.
    """
    
    def __init__(self, dataset_name : str, save_directory_path : str, file_name : str, dataset_size : int):
        
        self.dataset_name : str = dataset_name # "lucadiliello/english_wikipedia"
        self.save_directory_path : str = save_directory_path # "./wikipedia"
        self.file_name : str = file_name # english_wikipedia.hf
        self.load_full_filepath : str = f"../{self.save_directory_path}/{self.file_name}/" 
        self.dataset_size : int = int(dataset_size)
        self.dataset : Dataset = self.load_wikipedia_dataset()
        
    def load_wikipedia_dataset(self) -> Dataset:

        if os.path.isdir(f"{self.load_full_filepath}"):
            self.dataset = pd.read_csv(self.load_full_filepath).iloc[:,1:]
            return self.dataset
        else:
            
            dataset = load_dataset(self.dataset_name, "20231101.en", split="train", streaming=True)
            first_10k = dataset.take(self.dataset_size)
            first_10k = first_10k.remove_columns(['id','url'])
            data_list = list(first_10k)
            self.dataset = pd.DataFrame(data_list)
            self.save_dataset()
            return self.dataset

    def save_dataset(self) -> None:

        output_dir = f'../{self.save_directory_path}'

        if os.path.isdir(output_dir):
            self.dataset.to_csv(f"{output_dir}/{self.file_name}")
        else:
            os.mkdir(output_dir)
            self.dataset.to_csv(f"{output_dir}/{self.file_name}")
            
    def process_dataset(self) -> List:
        
            texts = []
            for row in self.dataset[['title', 'text']].itertuples(index=False):
                texts.append(row.title + " " + row.text)
            return texts
