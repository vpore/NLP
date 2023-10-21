import math
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

from nltk import word_tokenize
from collections import Counter

from lemmatization import lemmatization

ob = lemmatization()
corpus = ob.process(100)

tokenized_text = []
for text in corpus:
    words = word_tokenize(text)
    for word in words:
        tokenized_text.append(word)
        

bigram_list = list(nltk.bigrams(tokenized_text))
bigramCounts = Counter(nltk.bigrams(tokenized_text))
unigramCounts = Counter(tokenized_text)

print("\nN-GRAM :\n")
print("\nBigram frequency :\n", bigramCounts.most_common(20))
print("\nUnigram frequency :\n", unigramCounts.most_common(20))

bigram_freq = bigramCounts

def predict_next_word(word):
    next_word = None
    max_freq = 0

    for bigram in bigram_freq:
        if bigram[0] == word and bigram_freq[bigram] > max_freq:
            max_freq = bigram_freq[bigram]
            next_word = bigram[1]

    return next_word

input_word = input("Enter the word : ")
predicted_word = predict_next_word(input_word)
print(f"Predicted word after '{input_word}' : ", predicted_word)

bigram_prob = {}
for bigram in bigram_list:
    word1 = bigram[0]
    word2 = bigram[1]
    bigram_prob[bigram] = (bigramCounts.get(bigram))/(unigramCounts.get(word1))

test_corpus = ob.process(5)
for row in test_corpus:
    print(row)
    tword = []
    sample_text = row
    tword.extend(sample_text.split())
    testBiList = list(nltk.bigrams(tword))
    outputProb1 = 0
    for i in range(len(testBiList)):
        if testBiList[i] in bigram_prob:
            outputProb1 += math.log(bigram_prob[testBiList[i]])
        else:
            outputProb1 += 0
    print('Prob: ' + str(outputProb1))
    print()


print("\n\nTF-IDF :\n")

tfidf = TfidfVectorizer()

text = ' '.join(tokenized_text)
text = [text]
tfidf_matrix = tfidf.fit_transform(text)
feature_names = tfidf.get_feature_names_out()
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
df_trans = tfidf_df.transpose()
df_trans.reset_index(inplace=True)
df_trans.columns = ['Term', 'Frequency']
print(df_trans[20:30])
