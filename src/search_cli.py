import os
import argparse
from rag_retrieval_pipeline import RAGPipeline

def main():
    
    """ this is CLI which initilizes the RAGPipeline once and then provides the user to ask queries until they type "exit"
    """
    
    # Initialize the RAGPipeline (loaded once)
    print("Booting System")
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of search_cli.py
    config_path = os.path.join(current_dir, "../config/nested_configurations.yaml")
    
    rag_pipeline = RAGPipeline(config_path=config_path)
    print("RAGPipeline initialized successfully.")

    # Interactive CLI loop
    print("\nEnter your query to generate an answer. Type 'exit' to quit.")
    while True:
        # Take input from the user
        query = input("\nYour Query: ").strip()
        
        if query.lower() == 'exit':  # Exit condition
            print("Exiting the interactive session!")
            break
        
        if not query:
            print("Query cannot be empty. Please try again.")
            continue

        # Process the query using the RAGPipeline
        print("Processing User query...")
        try:
            response = rag_pipeline.generate_answer_with_rag(text_input=query)
            print("\n=== Generated Answer ===")
            print(response)
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()