# -*- coding: utf-8 -*-
"""
@author: Shubham
"""

import pycrfsuite
import numpy as np
from sklearn.metrics import classification_report
from bs4 import BeautifulSoup as bs
from bs4.element import Tag
import codecs
import nltk
from sklearn.model_selection import train_test_split

#Read data file and parse the XML
with codecs.open('reuters.xml', 'r', 'utf-8') as infile:
    soup = bs(infile, 'html5lib')
    
def contains_hyphen(word):
    if '-' in word:
        return True
    else:
        return False
    
def contains_underscore(word):
    if '_' in word:
        return True
    else:
        return False
    
docs = []
for element in soup.find_all('document'):
    texts = []
    
    for child in element.find('textwithnamedentities').children:
        if type(child) == Tag and child.name != None:
            if child.name == 'namedentityintext':
                label = 'NE'
            else:
                label = 'OE'
        
        if child.name != None:
            for w in child.text.split(' '):
                if len(w) > 0:
                    texts.append((w, label))
    
    docs.append(texts)

data = []
for i, doc in enumerate(docs):
    
    #Fetch the list of tokens in the document
    tokens = [t for t, label in doc]
    
    #Perform POS Tagging
    tagged = nltk.pos_tag(tokens)
    
    #Take the word, POS Tag and its label
    data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])
    
def word2features(doc, i):
    word = doc[i][0]
    postag = doc[i][1]
    
    #Common features for all words
    features = [
            'bias',
            'word.lower=' + word.lower(),
            'word[-3:]=' + word[-3:],
            'word[-2:]=' + word[-2:],
            'word.isupper=%s' % word.isupper(),
            'word.istitle=%s' % word.istitle(),
            'word.isdigit=%s' % word.isdigit(),
            'word.hyphen=%s' % contains_hyphen(word),
            'word.underscore=%s' % contains_underscore(word),
            'postag=' + postag
            ]
    
    #Features for words that are not at the beginning of a document
    if i > 0:
        word1 = doc[i-1][0]
        postag1 = doc[i-1][1]
        
        features.extend([
                '-1:word.lower=' + word1.lower(),
                '-1:word.isupper=%s' % word1.isupper(),
                '-1:word.istitle=%s' % word1.istitle(),
                '-1:word.isdigit=%s' % word1.isdigit(),
                '-1:word.hyphen=%s' % contains_hyphen(word1),
                '-1:word.underscore=%s' % contains_underscore(word1),
                '-1:postag=' + postag1
                ])
        
    else:
        #Indicate that it is the 'beginning of a document'
        features.append('BOS')
        
    #Features for words that are not at the end of a document
    if i < len(doc)-1:
        word1 = doc[i+1][0]
        postag1 = doc[i+1][1]
        
        features.extend([
                '+1:word.lower=' + word1.lower(),
                '+1:word.isupper=%s' % word1.isupper(),
                '+1:word.istitle=%s' % word1.istitle(),
                '+1:word.isdigit=%s' % word1.isdigit(),
                '+1:word.hyphen=%s' % contains_hyphen(word1),
                '+1:word.underscore=%s' % contains_underscore(word1),
                '+1:postag=' + postag1
                ])
        
    else:
        #Indicate that it is the 'end of a document'
        features.append('EOS')
        
    return features

#A function for extracting features in documents
def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]

# A function fo generating the list of labels for each document
def get_labels(doc):
    return [label for (token, postag, label) in doc]

X = [extract_features(doc) for doc in data]
y = [get_labels(doc) for doc in data]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

trainer = pycrfsuite.Trainer(verbose = True)

#Submit training data to the trainer
for x, y in zip(X_train, y_train):
    trainer.append(x, y)

#Set the parameters of the model 
trainer.set_params({
        #coefficient for L1 penalty
        'c1': 0.5,
        
        #coefficient for L2 penalty
        'c2': 0.05,
        
        #maximum number of iterations
        'max_iterations': 500,
        
        #whether to include transitions that are possible, but not observed
        'feature.possible_transitions': True
        })

trainer.train('crf.model')

tagger = pycrfsuite.Tagger()
tagger.open('crf.model')

y_pred = [tagger.tag(xseq) for xseq in X_test]

i = 5
for x, y in zip(y_pred[i], [x[1].split('=')[1] for x in X_test[i]]):
    print('%s (%s)' % (y, x))
    
#Create a mapping of labels to indices
labels = {"NE": 1, "OE": 0}

#Convert the sequences of tags into a 1-dimensional array  
predictions = np.array([labels[tag] for row in y_pred for tag in row])  
truths = np.array([labels[tag] for row in y_test for tag in row])
    
#Print out the classification report
print(classification_report(truths, predictions, target_names = ['I', 'N']))