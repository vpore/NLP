import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *

text = '''for a web application i am working on i would like to be able to give the user a single url that they can enter into the calendar application of their choosing to have events from our application show up in their calendar most other sites i have seen that do similar things will have a 1 time download of an ics file that can be imported if i have to require my users to download a new file every time the schedule changes it sort of defeats the purpose of having the feed at all the calendar can change many many times a day what i would really like is something like rss where their calendar program can look up a url and automatically see the most recent data does anything like this exist our main target is mobile devices so it really should be supported by ical and google calendar anything else is bonus'''

print("Text after preprocessing :\n", text)

words = word_tokenize(text)
stop_words = set(stopwords.words('english'))
words = [word for word in words if word not in stop_words]
text = ' '.join(words)
print("\nRemove Stop Words :\n", text)

nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
lemm_words = [token.lemma_ for token in doc]
lemm_text = ' '.join(lemm_words)
print("\nLemmatized Text :\n", lemm_text)

stemmer = PorterStemmer()
stem_words = [stemmer.stem(word) for word in words]
stem_text = ' '.join(stem_words)
print("\nStemmed Text :\n", stem_text)
