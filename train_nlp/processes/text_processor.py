import re
from nltk import PorterStemmer
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import warnings
import logging

warnings.filterwarnings('ignore')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')  

lemmatizer = WordNetLemmatizer()
stop_words = list(stopwords.words('english'))

ps = PorterStemmer()

def stemmerAndLemmitization(text:str):
    tokens = list(text.split(' '))
    tokens = [ps.stem(word) for word in tokens]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def textProcess(text):
    """ Filters the text with some regix expressions and apply stemming and lemmatization """
    try:
        logging.info(f"Enterd into textProcess with text = {text}")
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^A-Za-z\s]', '', text)
        text = text.lower()
        tokens = stemmerAndLemmitization(text)
        processed_text = ' '.join(tokens)
        return processed_text

    except Exception as e:
        logging.warning(f"could not perform text processing for {text}")
        return None