import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# nltk.download('punkt')
# nltk.download('stopwords')

import re

class preprocessing:

    def __init__(self) -> None:
        pass

    #Removes Punctuations
    def remove_punctuations(self, data):
        punct_tag=re.compile(r'[^\w\s]')
        data=punct_tag.sub(r'',data)
        return data

    #Removes HTML syntaxes
    def remove_html(self, data):
        html_tag=re.compile(r'<.*?>')
        data=html_tag.sub(r'',data)
        return data

    #Expand contractions
    def expand_cont(self, data):
        data = re.sub(r"he's", "he is", data)
        data = re.sub(r"i've", "i have", data)
        data = re.sub(r"can't", "can not", data)
        data = re.sub(r"they're", "they are", data)
        data = re.sub(r"let's", "let us", data)
        data = re.sub(r"it's", "it has", data)
        return data

    #Remove whitespace
    def remove_whitespace(self, data):
        data = re.sub(r'\s+',' ',data)
        return data

    #Removes URL data
    def remove_url(self, data):
        url_clean = re.compile(r"https://\S+|http://\S+|www\.\S+")
        data=url_clean.sub(r'',data)
        return data

    #Convert number-words to numbers
    def convert_to_numbers(self, data):

        words_to_numbers = {
            'one': '1',
            'two': '2',
            'three': '3',
            'four': '4',
            'five': '5',
            'six': '6',
            'seven': '7',
            'eight': '8',
            'nine': '9',
            'zero': '0'
        }

        pattern = re.compile(r'\b(' + '|'.join(words_to_numbers.keys()) + r')\b')
        return re.sub(pattern, lambda x: words_to_numbers[x.group()], data)

    def process(self, nrows=10, skiprows=[]):
        df = pd.read_csv('news.csv', skiprows=skiprows, nrows=nrows)
        # dataset - https://www.kaggle.com/datasets/pratul007/indian-express-scraped-dataset-for-last-one-year
        raw = list(df['Headline'])
        corpus = []
        stop_words = set(stopwords.words('english'))

        for text in raw:
            text = text.lower()
            text = self.remove_html(text)
            text = self.expand_cont(text)
            text = self.remove_url(text)
            text = self.remove_punctuations(text)
            text = self.convert_to_numbers(text)
            text = self.remove_whitespace(text)
            words = word_tokenize(text)
            words = [word for word in words if word not in stop_words]
            text = ' '.join(words)
            corpus.append(text)

        return corpus