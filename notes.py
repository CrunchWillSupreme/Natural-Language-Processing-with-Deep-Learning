# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 13:48:47 2020

@author: willh
"""
"""
Open files with plain text
open(filename, encoding='utf-8')

word embedding = a word vector

V x D = vocabulary size (# of total words) x vector dimensionality 
ex. if we're counting up how many times a word appears in a set of books, D = the total number of books

Word Analogies:
King - Man = Queen - Woman
King - Queen ~= Prince - Princess
France - Paris ~= Germany - Berlin

How to find analogies:
There are 4 words in every analogy
Input: 3 words
Output: find the 4th word
ex. King - Man = ? - Woman
Kng - Man + Woman = ?

closes_distance = infinity
best_word = None
test_vector = king - man + woman
for word, vector in vocabulary:
    distance = get_distance(test_vector, vector)
    if distance < closest_distance:
        closest_distance = distance
        best_word = word
        
Distances: 
Euclidean dist:
Cosine dist: most common