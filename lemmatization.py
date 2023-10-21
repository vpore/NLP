import spacy

from preprocessing import preprocessing

class lemmatization:
    def process(self, nrows=10, skiprows=[]):
        ob = preprocessing()
        raw = ob.process(nrows, skiprows)
        nlp = spacy.load('en_core_web_sm')

        corpus = []
        for text in raw:
            doc = nlp(text)
            lemm_words = [token.lemma_ for token in doc]
            lemmatized_text = ' '.join(lemm_words)
            corpus.append(lemmatized_text)

        return corpus
