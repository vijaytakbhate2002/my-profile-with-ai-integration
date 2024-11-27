import os

FAISS_DB_FILE = "faiss_document_store.db"
FAISS_INDEX_FOLDER = "faiss_index"
FAISS_CONFIG_FILE = os.path.join(FAISS_INDEX_FOLDER, "faiss_config.json")
FAISS_INDEX_FILE = os.path.join(FAISS_INDEX_FOLDER, "faiss_index")
CONTENT = "data\\resume_content1.txt"

RAG_MODEL = "deepset/minilm-uncased-squad2"


QUERY_EMBEDDING_MODEL= "sentence-transformers/all-mpnet-base-v2"
PASSAGE_EMBEDDING_MODEL= "sentence-transformers/all-mpnet-base-v2"
USE_GPU= True

TOP_K_RETRIEVER = 3
TOP_K_READER = 3