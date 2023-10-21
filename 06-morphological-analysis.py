import nltk
from polyglot.text import  Word
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

text = '''Morphological analysis is a field of linguistics that studies the structure of words. It identifies how a word is produced through the use of morphemes. A morpheme is a basic unit of the English language. The morpheme is the smallest element of a word that has grammatical function and meaning. Free morpheme and bound morpheme are the two types of morphemes. A single free morpheme can become a complete word.'''

tokens = word_tokenize(text)
tagged = nltk.pos_tag(tokens)

lemmatizer = WordNetLemmatizer()

for word, tag in tagged:
  lemma_word = lemmatizer.lemmatize(word)
  word = Word(word, language="en")
  print("{:<20}{:<5}{}".format(word, tag, word.morphemes))