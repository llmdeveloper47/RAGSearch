import os
import pinecone
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from openAI_module import OpenAIModule
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from transformers import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()
# Access the API keys
PINECONE_API_KEY = os.getenv("PINECONE")


class PineconeIndex:
    """ class to create and load an Index on PineCone free tier. 
        Arguments : 
            pinecone_api_key : the api key use to connect to Pinecone client
            index_name : the name of the index we wish to create
            embedding_model_name : the embedding model used to embed the documents
            dimension : the dimesion of the embedding
            
        We use a very simple moving window chunking strategy to create text chunks and embed the chunks in a document. Other possible options could be to use semantic chunking or langchain/llamaindex, however to build other components I had to stick with moving window approach
    """
    def __init__(self, config_path : str, index_name: str, embedding_model_name: str = 'BAAI/bge-large-en-v1.5', dimension: int = 1024, cloud: str = 'aws', region: str = 'us-east-1'):
        # Initialize Pinecone
        
        self.embedding_model_name = embedding_model_name
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(current_dir, "../config/nested_configurations.yaml")
    
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        
        pc = Pinecone(api_key=PINECONE_API_KEY)

        # Create or connect to a serverless Pinecone index
        self.index_name = index_name
        if index_name not in pc.list_indexes().names():
            print(f'creating index {index_name}')
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine',
                spec=ServerlessSpec(cloud=cloud, region=region)
            )
            self.index = pc.Index(index_name)
        else:
            print(f'Loading index {index_name}')
            self.index = pc.Index(index_name)

        # Initialize SentenceTransformer model for embeddings
        #self.embedding_model = SentenceTransformer(embedding_model_name, trust_remote_code=True)
        self.dimension = dimension
        #self.open_ai_object : OpenAIModule = OpenAIModule(config_path = self.config_path)
        self.embedding_model : SentenceTransformer = None
        self.load_model()

        
    def load_model(self, save_to_disk=True):
        """
        Load the SentenceTransformer model, either from disk or HuggingFace Hub.
        
        Parameters:
            model_path (str): Path to a locally saved model. If None, load from HuggingFace Hub.
            save_to_disk (bool): If True, save the model locally when loaded from the hub.
        """
        
        
        model_dir = f'../models/{self.embedding_model_name.replace("/", "_")}'
        
        if model_dir and os.path.exists(model_dir):
            print(f"Loading model from disk: {model_dir}")
            self.embedding_model = SentenceTransformer(model_dir, trust_remote_code=True)
        else:
            print(f"Loading model from HuggingFace Hub: {self.embedding_model_name}...")
            self.embedding_model = SentenceTransformer(self.embedding_model_name, trust_remote_code=True)
            if save_to_disk:
                # Save the model locally
                
                #print(f"Saving model locally to: {model_dir}")
                self.embedding_model.save(model_dir)    
                        
                
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        Split the text into chunks of specified size with overlap.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def embed_chunks(self, texts: List[str]) -> List[float]:
        """
        Generate embeddings for the input text using the specified SentenceTransformer model.
        """
        embeddings = self.open_ai_object.get_embeddings_chunks(texts = texts)
        return embeddings
    
    def embed_query(self, text : str) -> List[float]:
        
        embedding = self.open_ai_object.get_embedding_query(text = text)
        return embedding
    
    
    def embed_chunks_sentence_tranformers(self, texts: List[str]) -> List[float]:
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
        return embeddings
    

    def upsert_document(self, document_id: str, title: str, text: str):
        """
        Process the document by summarizing, chunking, embedding, and upserting into Pinecone.
        """
        # Combine title and text
        combined_text = f"{title}\n\n{text}"
        
        # Split the combined text into chunks
        chunks = self.chunk_text(combined_text)
        
        # Generate embeddings for all chunks
        embeddings = self.embed_chunks_sentence_tranformers(chunks)
        
        # Prepare data for upsert
        upsert_data = [
            (
                f"{document_id}_chunk_{i}",  # Unique ID for each chunk
                embedding,  # Embedding vector
                {
                    'document_id': document_id,
                    'chunk_index': i,
                    'title': title,
                    'chunk_text': chunk  # Include the chunk text in metadata
                }
            )
            for i, (embedding, chunk) in enumerate(zip(embeddings, chunks))
        ]

        # Upsert the embeddings into the Pinecone index
        self.index.upsert(vectors = upsert_data)

    def retrieve_documents(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        Query the Pinecone index with the provided text and return the top_k most similar chunks.
        """
        # Generate embedding for the query text
        query_embedding = self.embedding_model.encode(query_text)        
        
        # Query Pinecone index
        results = self.index.query(vector=query_embedding, top_k=top_k, include_values=False, include_metadata=True)

        # Extract and return metadata of the top_k results
        return [match.metadata for match in results.matches]