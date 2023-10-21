
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
# nltk.download('punkt')
# nltk.download('stopwords')

import re

#Removes Punctuations
def remove_punctuations(data):
    punct_tag=re.compile(r'[^\w\s]')
    data=punct_tag.sub(r'',data)
    return data

#Removes HTML syntaxes
def remove_html(data):
    html_tag=re.compile(r'<.*?>')
    data=html_tag.sub(r'',data)
    return data

#Expand contractions
def expand_cont(data):
    data = re.sub(r"he's", "he is", data)
    data = re.sub(r"i've", "i have", data)
    data = re.sub(r"can't", "can not", data)
    data = re.sub(r"they're", "they are", data)
    data = re.sub(r"let's", "let us", data)
    data = re.sub(r"it's", "it has", data)
    return data

#Remove whitespace
def remove_whitespace(data):
    data = re.sub(r'\s+',' ',data)
    return data

#Convert number-words to numbers
def convert_to_numbers(data):

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

# text = '''<p>For a web application I am working on, I would like to be able to give the user a single url that they can enter into the calendar application of their choosing to have events from our application show up in their calendar.</p>\n\n<p>Most other sites I have seen that do similar things will have a one time download of an .ics file that can be imported. If I have to require my users to download a new file every time the schedule changes, it sort of defeats the purpose of having the feed at all. The calendar can change many many times a day. </p>\n\n<p>What I would really like is something like rss where their calendar program can look up a url and automatically see the most recent data. Does anything like this exist? Our main target is mobile devices, so it really should be supported by iCal and google Calendar. Anything else is bonus.</p>\n'''


text = '''This is a sample text with some extra    spaces and special characters!
    It's got contractions like can't, I've, they're, he's.
    There are numbers like 12345 and words like one and two.
    also html tags like <p> </p>, <h1> </h1>, <ul> <li> </li> </ul>.
    Stopwords are common words such as the, and, in, to, with.
    Let's perform text processing on this text!
'''

print("Original Text :\n", text)

text = text.lower()

text = remove_html(text)
print("\nRemove html :\n", text)

sentences = sent_tokenize(text)

text = expand_cont(text)
print("\nExpand contractions :\n", text)

text = remove_punctuations(text)
print("\nRemove punctuations :\n", text)

text = convert_to_numbers(text)
print("\nConvert number words :\n", text)

text = remove_whitespace(text)
print("\nRemove whitespace :\n", text)

words = word_tokenize(text)
stop_words = set(stopwords.words('english'))
words = [word for word in words if word not in stop_words]
print("\nRemove Stop Words :\n", words)

# final_text = ' '.join(words)
# print("\nFinal text :\n", final_text)

print("\nNo. of Sentences :", len(sentences))
print("No. of words :", len(words))