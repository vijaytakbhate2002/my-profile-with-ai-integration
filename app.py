import os
import streamlit as st
import logging
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.FileHandler("logfile.log", mode='a')]
)
from processes import document_writer
from processes.document_writer import RAG
from processes.context_seperator import seperateContext
from processes.document_loader import DocumentLoader
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever
from config import config

class Retriever(RAG):

    def __init__(self, file_path:str, file_format:str, faiss_folder:str = config.FAISS_INDEX_FOLDER) -> None:
        super().__init__(file_path=file_path, file_format=file_format, faiss_folder=faiss_folder)


    def makeRetriever(self, document_store:FAISSDocumentStore) -> DensePassageRetriever:

        retriever = DensePassageRetriever(
            document_store=document_store,
            query_embedding_model=config.QUERY_EMBEDDING_MODEL,
            passage_embedding_model=config.PASSAGE_EMBEDDING_MODEL,
            use_gpu=config.USE_GPU
            )
        
        return retriever



    def fit(self) -> None:
        """" Load document from given path and write document to """

        os.makedirs(self.faiss_folder, exist_ok=True)
        document_store = FAISSDocumentStore(embedding_dim=768)
        retriever = self.makeRetriever(document_store=document_store)

        loader = DocumentLoader(file_path=self.file_path, 
                                file_format=self.file_format)
        documents = loader.loadFile()

        self.writeInDocumentStore(documents = documents, 
                                  document_store = document_store, 
                                  retriever=retriever)
        


    def retrieve(self, query: str, model_name:str) -> dict:
        """
        Run the pipeline and return retrieval results from the document store.
        """

        document_store = FAISSDocumentStore.load(
                                                index_path=os.path.join(config.FAISS_INDEX_FILE),
                                                config_path=os.path.join(config.FAISS_CONFIG_FILE)
                                                )
        retriever = self.makeRetriever(document_store=document_store)

        print("Creating Farm reader ......")
        self.reader = FARMReader(model_name_or_path=model_name) 
        self.pipeline = ExtractiveQAPipeline(reader=self.reader, retriever=retriever)
        results = self.pipeline.run(
            query=query,
            params={"Retriever": {"top_k": config.TOP_K_RETRIEVER}, "Reader": {"top_k": config.TOP_K_READER}}
        )

        return results


    def showAnswer(self, query: str, model_name:str="deepset/minilm-uncased-squad2") -> None:
        """
        Retrieve answers to a query and display them.
        """
        results = self.retrieve(query=query, model_name=model_name)
        return results



if __name__ == "__main__":

    file_path = config.CONTENT
    rag = Retriever(file_path=file_path, file_format='.txt')

    if os.path.exists(config.FAISS_CONFIG_FILE) and os.path.exists(config.FAISS_DB_FILE):
        print("Loading created embedings and db...")
    else:
        print("fitting document to generate embeddings...")
        rag.fit()

    model = config.RAG_MODEL
    query = input("Enter your question here...")
    results = rag.showAnswer(query=query, model_name=model)
    for result in results['answers']:
        print(result.answer)
        print("--------------")
       
