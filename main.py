import pandas as pd
import streamlit as st
import os
import logging
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
import PyPDF2
from processes.context_seperator import seperateContext

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.FileHandler("logfile.log", mode='a')]
)


class DocumentLoader:
    def __init__(self, file_path:str, file_format:str):
        self.file_path = file_path
        self.file_format = file_format

    def loadDocuments(self) -> list:
        """
        Reads a PDF file from the given path and extracts text as a list of document objects.
        """
        documents = []
        try:
            with open(self.file_path, "rb") as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                for page in reader.pages:
                    text = page.extract_text().strip()
                    if text:
                        documents.append({"content": text})
        except Exception as e:
            raise ValueError("Failed to load pdf check pdf path...")
        return documents
    

    def loadText(self) -> list:
        documents = seperateContext(self.file_path)
        return documents
    
    
    def loadFile(self) -> list[dict]:
        if '.txt' == self.file_format:
            return self.loadText()
        elif '.pdf' == self.format:
            return self.loadDocuments()
        
        else:
            raise ValueError("Invalid file format, DocumentLoader supports .txt and .pdf file format")



class RAG:

    config = {
        "query_embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "passage_embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "use_gpu": True
    }

    def __init__(self, file_path: str, file_format:str, faiss_folder: str = "faiss_index") -> None:
        """
        Initialize the RAG pipeline and reuse existing FAISS index folder if available.
        Args:
            file_path: Path to the input file.
            faiss_folder: Folder where FAISS index and metadata are stored.
        """
        self.file_path = file_path
        self.file_format = file_format
        self.faiss_folder = faiss_folder

        self.document_store = FAISSDocumentStore.load(index_path=os.path.join(self.faiss_folder, "faiss_index"),
                                                        config_path=os.path.join(self.faiss_folder, "faiss_config.json"))
        
        self.retriever = DensePassageRetriever(
            document_store=self.document_store,
            query_embedding_model=self.config["query_embedding_model"],
            passage_embedding_model=self.config["passage_embedding_model"],
            use_gpu=self.config["use_gpu"]
        )

        if not self.document_store.faiss_indexes:
            self.document_store.update_embeddings(self.retriever)


    def writeInDocumentStore(self) -> None:
        """
        Write documents to the document store and save FAISS index to folder.
        """
        
        loader = DocumentLoader(file_path=self.file_path, file_format=self.file_format)
        documents = loader.loadFile()

        if documents:
            existing_docs = len(self.document_store.get_all_documents())

            if existing_docs == 0:
                try:
                    self.document_store.write_documents(documents)
                    self.document_store.update_embeddings(self.retriever)
                except Exception as e:
                    raise ValueError("Error during write dicument and update embeddings.....")
                self.document_store.save(index_path=os.path.join(self.faiss_folder, "faiss_index"),
                                         config_path=os.path.join(self.faiss_folder, "faiss_config.json"))
            else:
                logging.info(f"Document store already contains {existing_docs} documents. Skipping write.")
        else:
            raise ValueError("No documents were loaded from the PDF.")



    def retrieve(self, query: str, model_name:str) -> dict:
        """
        Run the pipeline and return retrieval results from the document store.
        """

        self.reader = FARMReader(model_name_or_path=model_name)
        self.pipeline = ExtractiveQAPipeline(reader=self.reader, retriever=self.retriever)
        results = self.pipeline.run(
            query=query,
            params={"Retriever": {"top_k": 3}, "Reader": {"top_k": 3}}
        )
        return results



    def showAnswer(self, query: str, model_name:str) -> None:
        """
        Retrieve answers to a query and display them.
        """
        self.writeInDocumentStore()  
        results = self.retrieve(query=query, model_name=model_name)
        return results


if __name__ == "__main__":

    models = [
            # "deepset/roberta-base-squad2", # 0
            "deepset/minilm-uncased-squad2", # +1
            ]

    query = st.text_input("Enter your question here..")
    file_path = "RAG_model_pdf\\resume_content1.txt"
    rag = RAG(file_path=file_path, file_format='.txt')

    for model in models:
        results = rag.showAnswer(query=query, model_name=model)
        st.header(model, divider='rainbow')
        for result in results['answers']:
            st.write(result.answer)
            st.write("---------------")



