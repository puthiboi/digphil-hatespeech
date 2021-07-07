import pandas as pd
import numpy as np
from numpy import argmax
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.naive_bayes import MultinomialNB



### Load & Filter dataset
df = pd.read_csv("labeled_data.csv")
df_selected = df.drop(['Unnamed: 0','count', "offensive_language", "neither", 'class'], axis=1)

### convert remaining data to machine learning readable form
hate_speech = np.asarray(df_selected.hate_speech)
tweets = np.asarray(df_selected.tweet)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(tweets)
X_train_counts.shape

tfidf_transformer = TfidfTransformer()
tweets_tfidf = tfidf_transformer.fit_transform(X_train_counts)
tweets_tfidf.shape

### split the data
tweets_tfidf_train, tweets_tfidf_test, hate_speech_train, hate_speech_test = train_test_split(
    tweets_tfidf, hate_speech, 
    test_size=0.20, random_state=42)


### train the classifier using the training data

#clf = MultinomialNB()
#clf.fit(tweets_tfidf_train, hate_speech_train)

clf = RandomForestClassifier()
clf.fit(tweets_tfidf_train, hate_speech_train)

### compute accuracy using test data
acc_test = clf.score(tweets_tfidf_test, hate_speech_test)
print ("Test Accuracy:", acc_test)