#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:12:04 2018

original with POS tagging to make lemmatization more accurate run for 14 minutes. This versions runs for a minute

@author: matyask and mohammed hejazi
"""
import json, nltk, operator, csv
from nltk.corpus import words
from nltk.corpus import stopwords

# put stopwords and words into set for faster lookup
corpStopwords = set(stopwords.words('english'))
corpWords = set(words.words('en'))
# load lemmatizer so there is no need to load it each loop
lem = nltk.WordNetLemmatizer()

# global variable that will store review text and stars
reviewDict = {}
reviewKeys = set() # used for faster lookup of words

with open('yelp_academic_dataset_review_small.json') as inFile:
    jsonReviews = json.load(inFile)
    # for every Yelp review, process text and stars
    for review in jsonReviews:
        # wordpunct_tokenize runs for 2 minutes less than word_tokenize
        tempText = set(nltk.wordpunct_tokenize(review['text']))
        # turn to lowercase because corpus words are all lowercase
        tempText = set([lem.lemmatize(w).lower() for w in tempText])
        for word in tempText:
            if word in reviewKeys:
                # add stars from different review for future calculation
                reviewDict[word].append(review['stars'])
            # omit stopwords and only keep words in corpus words
            elif word in corpWords:
                if word in corpStopwords:
                    continue #no need to do anything if the word is in stopwords
                else:
                    reviewDict[word] = [review['stars']]
                    reviewKeys.add(word)

# calculate mean for words with more than 10 reviews
reviewTuples = [(key, sum(value)/len(value)) for key, value in reviewDict.items() if len(value) >= 10]
reviewTuples.sort(key=operator.itemgetter(1), reverse = True)
reviewTuples = reviewTuples[:500] + reviewTuples[-500:]
with open('topLemmmas.csv', 'w') as outFile:
    writer = csv.writer(outFile)
    writer.writerow(['Word','Sentiment'])
    writer.writerows(reviewTuples)
