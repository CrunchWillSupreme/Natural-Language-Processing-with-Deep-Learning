# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 20:57:01 2020

@author: willh
"""

### Build your own spam detector
"""
columns 1..48:
    word-frequency measure - number of times word appears divided by number of words in document x 100
Last column is a label
    1=spam, 0=not spam
One example of a "term-document matrix" - terms go along columns, documents (aka emails) go along rows
"""

from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
import os

os.chdir(r'C:\Users\willh\codes\Natural Language Processing with Deep Learning\nlp_class')

data = pd.read_csv('spambase.data').as_matrix() # using as_matrix b/c we don't need the pandas stuff
np.random.shuffle(data) # inplace shuffle of data

X = data[:, :48]
Y = data[:, -1]

Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print("Classification rate for NB:", model.score(Xtest, Ytest))

from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier()
model.fit(Xtrain, Ytrain)
print("Classification rate for AdaBoost:", model.score(Xtest, Ytest))

### spam detector 2
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

#data2 = pd.read_csv('../large_files/spam.csv', encoding = 'latin-1')
data2 = pd.read_csv('../large_files/spam.csv', encoding = 'ISO-8859-1')
# drop unnecessary columns
data2 = data2.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
# rename the columns
data2.columns = ['labels', 'data']

# create binary labels
data2['b_labels'] = data2['labels'].map({'ham':0, 'spam': 1})
Y = data2['b_labels'].as_matrix()

# try multiple ways of calculating features
# tfidf = TfidfVectorizer(decode_error='ignore')
# X = tfidf.fit_transform(df['data'])

count_vectorizer = CountVectorizer(decode_error='ignore')
X = count_vectorizer.fit_transform(data2['data'])

# split up the data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)

# create the model, train it, print scores
model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print("train score:", model.score(Xtrain, Ytrain))
print("test score:", model.score(Xtest, Ytest))

# visualize the data
def visualize(label):
    words = ''
    for msg in data2[data2['labels'] == label]['data']:
        msg = msg.lower()
        words += msg + ' '
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
visualize('spam')
visualize('ham')

# see what we're getting wrong
data2['predictions'] = model.predict(X)

# things that should be spam
sneaky_spam = data2[(data2['predictions'] == 0) & (data2['b_labels'] == 1)]['data']
for msg in sneaky_spam:
    print(msg)
    
# things that should not be spam
not_actually_spam = data2[(data2['predictions'] == 1) & (data2['b_labels'] == 0)]['data']
for msg in not_actually_spam:
    print(msg)