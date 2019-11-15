# -*- coding: utf-8 -*-
## NLP Text Classification
### Import the Libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk

nltk.download('all')

# quoting = 3 will ignore double quotes
dataset = pd.read_csv('Path to the dataset', delimiter = '\t', quoting = 3)

dataset.head()

# Sample sentence
dataset['Review'][0]

# Sample sentence
dataset['Review'][6]

"""### Import Stop Words"""

from nltk.corpus import stopwords

"""### Import Stemmer Class"""

from nltk.stem.porter import PorterStemmer

"""Instantiate the Stemmer"""

ps = PorterStemmer()

"""### Create a Corpus of clean text
Apply Regular expression , Stemming and Stopwords to get a corpus of clean words
"""

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

## Sample sentence after cleansing, stemming and applying stop words
corpus[0]

## Sample sentence after cleansing, stemming and applying stop words
corpus[6]


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 1500, min_df = 3, max_df = 0.6)


X = vectorizer.fit_transform(corpus).toarray()

# TF-IDF vector for sample sentences
X[0]

y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Training the KNN model
from sklearn.neighbors import KNeighborsClassifier
# minkowski is for ecledian distance
classifierKNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifierKNN.fit(X_train, y_train)


#from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
#classifier = GaussianNB()
classifierNB = MultinomialNB()
classifierNB.fit(X_train, y_train)

y_pred_knn = classifierKNN.predict(X_test)

y_pred_NB = classifierNB.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score,confusion_matrix

cmknn = confusion_matrix(y_test, y_pred_knn)
cmknn

cmNB = confusion_matrix(y_test, y_pred_NB)
cmNB

print("KNN accuracy \n", accuracy_score(y_test,y_pred_knn))

print("Naive Bayes accuracy \n", accuracy_score(y_test,y_pred_NB))


sample = ["Good batting by England"]

# create the TF-IDF model of the sample sentence
sample = vectorizer.transform(sample).toarray()

#predict the sentiment
sentiment = classifierNB.predict(sample)
if (sentiment==1):
    print("Good Review")
else:
    print("Bad Review")

sample2 = ["bad performance by India in the match"]
sample2 = vectorizer.transform(sample2).toarray()
sentiment2 = classifierNB.predict(sample2)
if (sentiment2==1):
    print("Good Review")
else:
    print("Bad Review")