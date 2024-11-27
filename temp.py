# from haystack.document_stores import FAISSDocumentStore
# from haystack.nodes import DensePassageRetriever
# import PyPDF2
# from haystack.nodes import FARMReader
# from haystack.pipelines import ExtractiveQAPipeline
# import logging

# logging.basicConfig(
#     level=logging.DEBUG,  
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
#     datefmt='%Y-%m-%d %H:%M:%S',  
#     handlers=[
#         logging.StreamHandler(),  
#         logging.FileHandler("logfile.log", mode='w') 
#     ]
# )


# class RAG:

#     config = {
#         "model_name_or_path":"deepset/roberta-base-squad2",
#         'query_embedding_model':"sentence-transformers/all-mpnet-base-v2",
#         'passage_embedding_model': "sentence-transformers/all-mpnet-base-v2",
#         'use_gpu':True
#     }

#     def __init__(self, pdf_path:str) -> None:
#         """ Args: pdf_path

#             This function helps to assign pdf_path, document_store, retriever, reader and pipeline

#             Return: None"""

#         self.pdf_path = pdf_path
#         self.document_store = FAISSDocumentStore(embedding_dim=768) 

#         self.retriever = DensePassageRetriever(
#                     document_store=self.document_store,
#                     query_embedding_model=self.config['query_embedding_model'],
#                     passage_embedding_model=self.config["passage_embedding_model"],
#                     use_gpu=self.config['use_gpu']
#                     )

#         self.reader = FARMReader(model_name_or_path=self.config["model_name_or_path"])
#         self.pipeline = ExtractiveQAPipeline(reader=self.reader, retriever=self.retriever)
        


#     def loadDocuments(self) -> None:
#         """
#         Reads a PDF file from the given path and extracts text as a list of document objects.
#         Args:
#             file_path (str): Path to the PDF file.
#         Returns:
#             list: List of dictionaries containing extracted content.
#         """
#         documents = []
#         try:
#             with open(self.pdf_path, 'rb') as pdf_file:
#                 reader = PyPDF2.PdfReader(pdf_file)
#                 for page in reader.pages:
#                     text = page.extract_text().strip()

#                     if text: 
#                         documents.append({"content": text})
#         except Exception as e:
#             print(f"Error reading PDF file: {e}")
#         logging.info(f"document successfully loaded... {documents}")
#         return documents



#     def writeInDocumentStore(self) -> None:
#         """ Write document in document store, update retriver in document store"""

#         documents = self.loadDocuments()
#         if documents:
#             self.document_store.write_documents(documents)
#             self.document_store.update_embeddings(self.retriever)
#         else:
#             print("No documents were loaded from the PDF.")



#     def retrive(self, query) -> None:
#         """ Run the pipeline and return result (retrival results from document) """
#         self.writeInDocumentStore()

#         results = self.pipeline.run(
#                                     query=query, 
#                                     params={"Retriever": {"top_k": 3}, 
#                                     "Reader": {"top_k": 1}}
#                                     )

#         return results

    

#     def showAnswer(self, query) -> None:

#         results = self.retrive(query=query)
#         print(results)
#         # for answer in results["answers"]:
#         #     print(f"Answer: {answer.answer} (Score: {answer.score})")

#         print(results["answers"])




# pdf_path = "RAG_model_pdf\Vijay_Takbhate_RAG_Model_Document_Cleaned.pdf"  

# rag = RAG(pdf_path=pdf_path)

# rag.showAnswer(query="Hey give me project details")

