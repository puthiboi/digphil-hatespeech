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
#from sklearn.naive_bayes import MultinomialNB


cores = 3 # how many cpu cores one wants to use (for most classifiers)

randforclf = False ##use the random forest classifier in this run?
no_of_trees = 240 #number of trees in the forest of the random forest classifier


adaboost = False #use adaboost in this run?
no_est_ada = 100 #number of weak estimators used for adaboost

baggingclf = False
no_est_bag = 100

extratrees = False #extra trees classifier, higher bias but lower variance than random forest
no_est_ext = 100

histgradboost = False #findet er irgendwie nicht im  modul
no_est_hgb = 100

gradboostclf = True #gradient boosting classifier
no_est_grb = 100

stackingclf = True
#no_est_sta = 100 not possible for stacking clf

print("loadfilter")
### Load & Filter dataset
df = pd.read_csv("labeled_data.csv")
df_selected = df.drop(['Unnamed: 0','count', "offensive_language", "neither", 'class'], axis=1)

print("convert")
### convert remaining data to machine learning readable form
hate_speech = np.asarray(df_selected.hate_speech)
tweets = np.asarray(df_selected.tweet)

print("countvec")
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(tweets)
X_train_counts.shape

print("tfidftrans")
tfidf_transformer = TfidfTransformer()
tweets_tfidf = tfidf_transformer.fit_transform(X_train_counts)
tweets_tfidf.shape

print("split")
### split the data
tweets_tfidf_train, tweets_tfidf_test, hate_speech_train, hate_speech_test = train_test_split(
    tweets_tfidf, hate_speech, 
    test_size=0.20, random_state=42)


### train the classifier using the training data

#clf = MultinomialNB()
#clf.fit(tweets_tfidf_train, hate_speech_train)

if randforclf == True:

    print(f"training Random Forest Classifier with {no_of_trees} estimators")
    clf = RandomForestClassifier(n_estimators=no_of_trees, n_jobs=cores)
    clf.fit(tweets_tfidf_train, hate_speech_train)

    #print("accTest")
### compute accuracy using test data
    acc_test = clf.score(tweets_tfidf_test, hate_speech_test)
    print ("Test Accuracy:", acc_test)

if adaboost == True:
    print(f"training AdaBoost with {no_est_ada} estimators")
    clf = AdaBoostClassifier(n_estimators=no_est_ada)  # , n_jobs=cores)
    clf.fit(tweets_tfidf_train, hate_speech_train)

    #print("accTest")
    ### compute accuracy using test data
    acc_test = clf.score(tweets_tfidf_test, hate_speech_test)
    print("Test Accuracy:", acc_test)

if baggingclf == True:
    print(f"training Bagging Classifier with {no_est_bag} estimators")
    clf = ens.BaggingClassifier(n_estimators=no_est_bag,n_jobs=cores)  # , n_jobs=cores)
    clf.fit(tweets_tfidf_train, hate_speech_train)

    # print("accTest")
    ### compute accuracy using test data
    acc_test = clf.score(tweets_tfidf_test, hate_speech_test)
    print("Test Accuracy:", acc_test)

if extratrees == True:
    print(f"training extra trees classifier with {no_est_ext} estimators")
    clf = ens.ExtraTreesClassifier(n_estimators=no_est_ext,n_jobs=cores)  # , n_jobs=cores)
    clf.fit(tweets_tfidf_train, hate_speech_train)

    # print("accTest")
    ### compute accuracy using test data
    acc_test = clf.score(tweets_tfidf_test, hate_speech_test)
    print("Test Accuracy:", acc_test)

if histgradboost == True:
    print(f"training HistGradBoostCLF with {no_est_hgb} estimators")
    clf = ens.HistGradientBoostingClassifier(n_estimators=no_est_hgb)  # , n_jobs=cores)
    clf.fit(tweets_tfidf_train, hate_speech_train)

    # print("accTest")
    ### compute accuracy using test data
    acc_test = clf.score(tweets_tfidf_test, hate_speech_test)
    print("Test Accuracy:", acc_test)

if gradboostclf == True:
    print(f"training Gradient Boosting Classifier with {no_est_grb} estimators")
    clf = ens.GradientBoostingClassifier(n_estimators=no_est_grb)  # , n_jobs=cores)
    clf.fit(tweets_tfidf_train, hate_speech_train)

    # print("accTest")
    ### compute accuracy using test data
    acc_test = clf.score(tweets_tfidf_test, hate_speech_test)
    print("Test Accuracy:", acc_test)

if stackingclf == True:
    print(f"training Stacking Classifier")
    clf = ens.StackingClassifier(n_estimators=no_est_sta,n_jobs=cores)  # , n_jobs=cores)
    clf.fit(tweets_tfidf_train, hate_speech_train)

    # print("accTest")
    ### compute accuracy using test data
    acc_test = clf.score(tweets_tfidf_test, hate_speech_test)
    print("Test Accuracy:", acc_test)
