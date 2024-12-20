from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever
import logging
import os
import joblib

class RAG():


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

    def writeInDocumentStore(self, documents:list, document_store:FAISSDocumentStore, retriever:DensePassageRetriever) -> None:
        """
        Write documents to the document store and save FAISS index to folder.
        """

        if documents:
            existing_docs = len(document_store.get_all_documents())

            if existing_docs == 0:
                logging.info("Writing documents to the document store...")
                try:
                    document_store.write_documents(documents)
                    document_store.update_embeddings(retriever)
                except Exception as e:
                    logging.info("Error during write dicument and update embeddings.....")
                logging.info("Documents and embeddings updated.")
                document_store.save(index_path=os.path.join(self.faiss_folder, "faiss_index"),
                                         config_path=os.path.join(self.faiss_folder, "faiss_config.json"))
            else:
                logging.info(f"Document store already contains {existing_docs} documents. Skipping write.")
        else:
            logging.warning("No documents were loaded from the PDF.")


