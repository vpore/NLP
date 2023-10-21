import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from nltk import pos_tag
from nltk.corpus import wordnet

df = pd.read_csv('fifa_world_cup_2022_tweets.csv')

print('2022 Fifa worldcup tweets dataset with the shape of', df.shape)
print(df.sample(5))
print()

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub(r'(^|\s)@(\w+)', r'\1@user', text)
    text = re.sub(r'\bhttps?://\S+\b', 'http', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    text = ' '.join(tokens)

    # POS tagging
    split_text = text.split()
    POS_tag = pos_tag(split_text)

    # WordNet analysis
    for word in split_text:
        if (wordnet.synsets(word) == []):
            continue
        syn = wordnet.synsets(word)[0]

    return text

robertahuggingface = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(robertahuggingface)
tokenizer = AutoTokenizer.from_pretrained(robertahuggingface)

def polarity_scores_roberta(Input):
    
    encoded_text = tokenizer(Input, return_tensors='pt')
    
    output = model(**encoded_text)
    
    scores = output[0][0].detach().numpy()
    
    scores = softmax(scores)
    
    return scores


df2 = df.head(500)
df3 = df.head(5)
df3['preprocessed_tweet'] = df3['Tweet'].apply(preprocess)

res = []
for i, row in tqdm(df2.iterrows(), total=len(df2)):
  text = row['Tweet']
  res.append(polarity_scores_roberta(text))

roberta = []
for i in range(len(res)):
  argmax = np.argmax(res[i])
  if argmax == 0:
    roberta.append('negative')
  elif argmax == 1:
    roberta.append('neutral')
  else:
    roberta.append('positive')
df2['roberta'] = roberta

print('sentiment column\n',df2['Sentiment'].value_counts().sort_index(),'\n---')

print('roberta model prediction\n',df2['roberta'].value_counts().sort_index())

sns.countplot(data=df2, x='roberta')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
