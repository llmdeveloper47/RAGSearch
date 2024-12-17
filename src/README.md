### INSTRUCTION : Running the Search CLI (locally)
To test the application locally, we follow the below steps:
Step1 - cd into src folder, 
Step2 - run '''python3 search_cli.py''' which will then initialize the Pinecone client to connect to the vector database for retrieval,Cohere client for reranking of retrieved documents and OpenAI for generation of the final answer

NOTE : For the purpose of testing this application, I have generated temporary API keys for OpenAI, Cohere and Pinecone.

NOTE : To save time, the index has been created for the first 10000 rows of the wikipedia dataset. More details about the dataset can be found under config/nested_configuration.yaml under section huggingface. However if we wish to create the index from scratch we can follow the below instructions to generate the index.

### INSTRUCTION  : Dataset Index Creation
To Build index from scratch please run the following script '''python3 index_builder.py''' , which using dataset_builder will first download the wikipedia dataset as specified in the config/nested_configuration.yaml file. Then create a subset of the dataset ( first 10,000 rows ) and save it in the root folder and then create a Pinecone Index for the selected records. 

As I'm using the free tier from Pinecone, we need to keep the storage and write units in check and hence sample the dataset to test the application.

### INSTRUCTION : Running the evaluation_data_generator.py

The script loads the dataset module and samples 100 documents from it, Then calls the openai module to generate 5 query - answer pairs per document and saves the results (sample_query_answer_pairs.json) in the evaluation folder. The saved file is used for load testing the application ( more on that in README.under tests folder)






