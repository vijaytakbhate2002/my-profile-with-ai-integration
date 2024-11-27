def seperateContext(text_file_path:str="resume_content.txt", seperator:str='-----') -> list[dict]:
    """ Args: text_file_path (string input of text file path)

        function splits text with given seperator and generate list of documents,
        documt format = [{'content':"firsrt splitted text from text file"}, 
                         {'content':"second splitted text from text file"}]

        Return: list[dict]"""
    
    with open(text_file_path, 'r') as file:
        data = file.read()
    sections = data.split(seperator)

    documents = []
    for section in sections:
        content = {'content':''}
        content['content'] = section
        documents.append(content)

    return documents



if __name__ == "__main__":
    result = seperateContext()
    print("number of chunks found = ", len(result))
    print(result)