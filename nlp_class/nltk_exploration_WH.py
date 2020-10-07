# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 14:12:36 2020

@author: willh
"""
%matplotlib auto # to pop images out of spyder
%matplotlib inline # to show images in spyder console
import nltk

# NN = noun, VBZ = verb 3rd person present, JJ = adjective
nltk.pos_tag('Machine learning is great'.split())

# Stemming and Lemmatization
# stemming is more crude
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
porter_stemmer.stem('wolves')
# returns 'wolv' - not a word

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize('wolves')
# returns 'wolf'

s = "Albert Einstein was born on March 14, 1879"
tags = nltk.pos_tag(s.split())
print(tags)

# Named entity recognizer
nltk.ne_chunk(tags)
nltk.ne_chunk(tags).draw()

s = "Steve Jobs was the CEO of Apple Corp."
tags = nltk.pos_tag(s.split())
nltk.ne_chunk(tags)
