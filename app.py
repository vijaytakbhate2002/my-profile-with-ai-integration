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


class Retriever(RAG):

    def __init__(self, file_path:str, file_format:str) -> None:
        super().__init__(file_path=file_path, file_format=file_format, faiss_folder='faiss_index')


    def fit(self) -> None:
        """" """
        loader = DocumentLoader(file_path=self.file_path, file_format=self.file_format)
        documents = loader.loadFile()
        self.writeInDocumentStore(documents=documents)


    def retrieve(self, query: str, model_name:str) -> dict:
        """
        Run the pipeline and return retrieval results from the document store.
        """

        print("Creating Farm reader ......")
        self.reader = FARMReader(model_name_or_path=model_name)
        self.pipeline = ExtractiveQAPipeline(reader=self.reader, retriever=self.retriever)
        results = self.pipeline.run(
            query=query,
            params={"Retriever": {"top_k": 3}, "Reader": {"top_k": 3}}
        )
        return results


    def showAnswer(self, query: str, model_name:str="deepset/minilm-uncased-squad2") -> None:
        """
        Retrieve answers to a query and display them.
        """
        results = self.retrieve(query=query, model_name=model_name)
        return results


if __name__ == "__main__":

    file_path = "data\\resume_content1.txt"
    rag = Retriever(file_path=file_path, file_format='.txt')

    if os.path.exists("faiss_index\\faiss_config.json") and os.path.exists("faiss_document_store.db"):
        print("Loading created embedings and db...")
    else:
        print("fitting document to generate embeddings...")
        rag.fit()

    model = "deepset/minilm-uncased-squad2"
    query = input("Enter your question here...")
    results = rag.showAnswer(query=query, model_name=model)
    for result in results['answers']:
        print(result.answer)
        print("--------------")
       
