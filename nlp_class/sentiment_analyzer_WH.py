# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 23:41:43 2020

@author: willh

To download nltk data - open ipython console:
    >>>import nltk
    >>>nltk.download()
    nltk downloader should open up.  Go to Packages and download necessary packages
"""

import nltk
import numpy as np
import os

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup #XML parser
os.getcwd()

wordnet_lemmatizer = WordNetLemmatizer() #turns words to their base form
stopwords = set(w.rstrip() for w in open('stopwords.txt'))

positive_reviews = BeautifulSoup(open('electronics/positive.review').read())
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open('electronics/negative.review').read())
negative_reviews = negative_reviews.findAll('review_text')

# we already know the number of positive reviews > # of negative reviews
np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[:len(negative_reviews)]


def my_tokenizer(s):
    s = s.lower() # downcase
    tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2] # remove any words that are less than 3 characters, they're probably not useful
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    return tokens

word_index_map = {}
current_index = 0


positive_tokenized = []
negative_tokenized = []

# Lemmatize and Tokenize the words in the reviews
for review in positive_reviews:
    tokens = my_tokenizer(review.text)
    positive_tokenized.append(tokens)
    
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1
            
#list(word_index_map.items())[:3]

for review in negative_reviews:
    tokens = my_tokenizer(review.text)
    negative_tokenized.append(tokens)
    
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1
            

# Create the term vector - Instead of using CountVectorizer or TfidfVectorizer
def token_to_vector(tokens, label):
    x = np.zeros(len(word_index_map) + 1) # add 1 for the label
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    x = x/x.sum()
    x[-1] = label
    return x

N = len(positive_tokenized) + len(negative_tokenized)

data = np.zeros((N, len(word_index_map) + 1))
i = 0 
for tokens in positive_tokenized:
    xy = token_to_vector(tokens, 1) # 1 is the label for positive reviews
    data[i,:] = xy
    i += 1
    
for tokens in negative_tokenized:
    xy = token_to_vector(tokens, 0) # 0 is the label for negative reviews
    data[i,:] = xy
    i += 1
    
np.random.shuffle(data)

X = data[:, :-1]
Y = data[:, -1]

Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

model = LogisticRegression()
model.fit(Xtrain, Ytrain)
print("Classification rate:", model.score(Xtest, Ytest))



threshold = 0.5
for word, index in word_index_map.items():
    weight = model.coef_[0][index]
    if weight > threshold or weight < -threshold:
        print(word, weight)