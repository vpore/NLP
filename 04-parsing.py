from spacy import displacy
import spacy
import nltk
from nltk import pos_tag
from nltk import RegexpParser

# Shallow parsing, or chunking, is the process of extracting phrases from unstructured text. This involves chunking groups of adjacent tokens into phrases on the basis of their POS tags. There are some standard well-known chunks such as noun phrases, verb phrases, and prepositional phrases.

text = '''Dependency parsing is the process of extracting the dependency graph of a sentence to represent its grammatical structure. It defines the dependency relationship between headwords and their dependents.'''
print("Original text :\n", text)

split_text = text.split()
POS_tag = pos_tag(split_text)
print("\nPOS tags :\n", POS_tag)

pattern = '''chunk: {<DT>?<JJ>*<NN>+|<DT>?<NNP>*<NN>*}'''
chunker = RegexpParser(pattern)
print("\nRegex Pattern :\n", pattern)

output = chunker.parse(POS_tag)
print("\nAfter Chunking :")
for subtree in output.subtrees():
    if subtree.label() == 'chunk':
        keyword = ' '.join(word for word, pos in subtree.leaves())
        print(keyword)
output.draw()



nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
print("\nDeep Parsing :")
for token in doc:
    print(f"""
        TOKEN : {token.text}
        {token.tag_ = }
        {token.dep_ = }"""
    )
displacy.serve(doc, style="dep")
