from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever
import logging
import os
import joblib

class RAG():

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

        if os.path.exists(self.faiss_folder + "/faiss_config.json"):
            logging.info("Loading existing FAISS document store...")
            self.document_store = FAISSDocumentStore.load(index_path=os.path.join(self.faiss_folder, "faiss_index"),
                                                          config_path=os.path.join(self.faiss_folder, "faiss_config.json"))
        else:
            logging.info("Creating a new FAISS document store...")
            os.makedirs(self.faiss_folder, exist_ok=True)
            self.document_store = FAISSDocumentStore(embedding_dim=768)

        self.retriever = DensePassageRetriever(
            document_store=self.document_store,
            query_embedding_model=self.config["query_embedding_model"],
            passage_embedding_model=self.config["passage_embedding_model"],
            use_gpu=self.config["use_gpu"]
        )

        if not self.document_store.faiss_indexes:
            logging.info("Updating embeddings for the new FAISS document store...")
            self.document_store.update_embeddings(self.retriever)


    def writeInDocumentStore(self, documents:list) -> None:
        """
        Write documents to the document store and save FAISS index to folder.
        """

        if documents:
            existing_docs = len(self.document_store.get_all_documents())

            if existing_docs == 0:
                logging.info("Writing documents to the document store...")
                try:
                    self.document_store.write_documents(documents)
                    self.document_store.update_embeddings(self.retriever)
                except Exception as e:
                    logging.info("Error during write dicument and update embeddings.....")
                logging.info("Documents and embeddings updated.")
                self.document_store.save(index_path=os.path.join(self.faiss_folder, "faiss_index"),
                                         config_path=os.path.join(self.faiss_folder, "faiss_config.json"))
            else:
                logging.info(f"Document store already contains {existing_docs} documents. Skipping write.")
        else:
            logging.warning("No documents were loaded from the PDF.")


