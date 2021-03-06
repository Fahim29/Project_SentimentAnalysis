# -*- coding: utf-8 -*-
"""SentimentAnalysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Q04l_jfjVj2ErQBQ_NcEI6XdvNSCkbBK
"""

# some necessary imports
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt

import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('popular')

train_df_sentiment = pd.read_csv('https://raw.githubusercontent.com/Fahim29/Project_SentimentAnalysis/master/Project/train-balanced-sentiment.csv',dtype=object)

train_df_sarcasm = pd.read_csv("G:\\@FinalYearProjects\\Project_SentimentAnalysis\\Project\\train-balanced-sarcasm.csv")


train_df_sentiment.dropna(subset=['review'], inplace=True)

train_df_sentiment['sentiment'].value_counts()

lm = WordNetLemmatizer()

corpus = []
for i in range(0, len(train_df_sentiment)):
    review = re.sub('[^a-zA-Z]', ' ', train_df_sentiment['review'][i])
    review = review.lower()
    review = review.split()
    review = [lm.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

y=pd.get_dummies(train_df_sentiment['sentiment'])
y=y.iloc[:,1].values

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier
sent_detect_model = MultinomialNB().fit(X_train, y_train)

sentiment=sent_detect_model.predict(X_test)

accuracy_score(y_test,sentiment)

train_df_sarcasm.head()

train_df_sarcasm.info()

train_df_sarcasm.dropna(subset=['comment'], inplace=True)

train_df_sarcasm['label'].value_counts()

corpus = []
for i in range(0, len(train_df_sarcasm)):
    review = re.sub('[^a-zA-Z]', ' ', train_df_sarcasm['comment'][i])
    review = review.lower()
    review = review.split()
    review = [lm.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
sarcasm_cv = CountVectorizer(max_features=2500)
sacasmX = sarcasm_cv.fit_transform(corpus).toarray()

sarcasmY=pd.get_dummies(train_df_sarcasm['label'])
sarcasmY=sarcasmY.iloc[:,1].values

# Train Test Split
from sklearn.model_selection import train_test_split
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(sacasmX, sarcasmY, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
sarc_detect_model = MultinomialNB()
sarc_detect_model.fit(X_train_s, y_train_s)

sarcasm=sarc_detect_model.predict(X_test_s)

accuracy_score(y_test_s,sarcasm)

positive_review = 0
neutral_review = 0
negative_review = 0
array = []

for i in range(0, len(X_test)):
    sentiment = sent_detect_model.predict(X_test)  
    sarcasm = sarc_detect_model.predict(X_test)
    if sentiment[0] == 1 and sarcasm[0] == 0 :
      positive_review += 1
      array.append(1)
    elif sentiment[0] == 0 and sarcasm[0] == 0 :
      negative_review += 1
      array.append(0)
    elif sentiment[0] == 1 and sarcasm[0] == 1 :
      negative_review += 1
      array.append(0)
    elif sentiment[0] == 0 and sarcasm[0] == 1 :
      positive_review += 1
      array.append(1)
    else:
      neutral_review += 1
      array.append(2)

positive_review

negative_review

neutral_review

accuracy_score(y_test,array)