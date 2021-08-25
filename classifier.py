import pandas as pd
import numpy as np
from numpy import argmax
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import sklearn.ensemble as ens
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
#from sklearn.naive_bayes import MultinomialNB


cores = 3 # how many cpu cores one wants to use (for most classifiers)

randforclf = False ##use the random forest classifier in this run?
no_of_trees = 240 #number of trees in the forest of the random forest classifier

adaboost = True #use adaboost in this run?
no_est_ada = 140 #number of weak estimators used for adaboost

gradboostclf = False
no_est_grb = 200

dropBots = True ## whether bots should be filtered out or not

#stackingclf = False
#no_est_sta = 100

print("loadfilter")
### Load & Filter dataset
df = pd.read_csv("labeled_data.csv")
df_selected = df.drop(['Unnamed: 0','count', "offensive_language", "neither", 'class'], axis=1)

print("convert")
### convert remaining data to machine learning readable form

##filtering out non-chars and converting tweets to lowercase
df_selected["tweet"] = df_selected["tweet"].apply(lambda x: (re.sub("[^a-zA-Z]"," ",x)).lower())
#print(df_selected["tweet"])


hate_speech = df_selected.hate_speech
tweets = df_selected.tweet

#print("countvec")
#count_vect = CountVectorizer()
#X_train_counts = count_vect.fit_transform(tweets)
#X_train_counts.shape

#print("tfidftrans")
#tfidf_transformer = TfidfTransformer()
#tweets_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#tweets_tfidf.shape

print("split")
### split the data
tweets_tfidf_train, tweets_tfidf_test, hate_speech_train, hate_speech_test = train_test_split(
    tweets, hate_speech,
    test_size=0.20, random_state=42)

tfVec = TfidfVectorizer(stop_words="english",ngram_range=(1,2),sublinear_tf=True)
tfVec.fit(tweets_tfidf_train)


if randforclf == True:

    print(f"training Random Forest Classifier with {no_of_trees} estimators")
    clf = RandomForestClassifier(n_estimators=no_of_trees, n_jobs=cores)
    clf.fit(tfVec.transform(tweets_tfidf_train), hate_speech_train)

    #print("accTest")
### compute accuracy using test data
    acc_test = clf.score(tfVec.transform(tweets_tfidf_test), hate_speech_test)
    print ("Test Accuracy:", acc_test)

if adaboost == True:
    print(f"training AdaBoost with {no_est_ada} estimators")
    clf = AdaBoostClassifier(n_estimators=no_est_ada)  # , n_jobs=cores)
    clf.fit(tfVec.transform(tweets_tfidf_train), hate_speech_train)

    #print("accTest")
    ### compute accuracy using test data
    acc_test = clf.score(tfVec.transform(tweets_tfidf_test), hate_speech_test)
    print("Test Accuracy:", acc_test)

if gradboostclf == True:
    print(f"training Gradient Boosting Classifier with {no_est_grb} estimators")
    clf = ens.GradientBoostingClassifier(n_estimators=no_est_grb)  # , n_jobs=cores)
    clf.fit(tweets_tfidf_train, hate_speech_train)

    # print("accTest")
    ### compute accuracy using test data
    acc_test = clf.score(tweets_tfidf_test, hate_speech_test)
    print("Test Accuracy:", acc_test)

## make a prediction using the last instance of clf
    #data prep

esc_df = pd.read_csv("eurosong.csv",encoding='utf-8',sep=',',lineterminator='\n')
soc_df = pd.read_csv("eurosoc.csv",encoding='utf-8',sep=',',lineterminator='\n')

#if dropBots == True:
 #   print("dropping bots...")
  #  for i,item in enumerate(esc_df.columns(["Username"])):
   #     if item == "ESC_bot_":
    #        esc_df.drop(esc_df[i])


esc_df = esc_df.drop(['Datetime',"Username"], axis=1)
esc_df = esc_df["Text"].apply(lambda x: (re.sub("[^a-zA-Z]"," ",x)).lower())

soc_df = soc_df.drop(["Datetime","Username"],axis = 1)
soc_df = soc_df["Text"].apply(lambda x: (re.sub("[^a-zA-Z]"," ",x)).lower())

#esc_ar = np.asarray(esc_df.Text)
#soc_ar = np.asarray(soc_df.Text)


    #predict

prediction = clf.predict(soc_df)
print(prediction)

