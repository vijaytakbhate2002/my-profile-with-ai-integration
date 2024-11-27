import PyPDF2
from processes.context_seperator import seperateContext
import logging

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
            logging.error(f"Error reading PDF file: {e}")
        logging.info(f"Documents successfully loaded: {documents}")
        return documents
    

    def loadText(self) -> list:
        documents = seperateContext(self.file_path)
        return documents
    
    
    def loadFile(self) -> list[dict]:
        if '.txt' == self.file_format:
            logging.info("Reading text document")
            return self.loadText()
        elif '.pdf' == self.format:
            logging.info("Reading text document")
            return self.loadDocuments()
        
        else:
            raise ValueError("Invalid file format, DocumentLoader supports .txt and .pdf file format")

