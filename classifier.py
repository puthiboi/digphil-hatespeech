import pandas as pd
import numpy as np
import time
#import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

###Set up the classifier 

cores = 3 # how many cpu cores one wants to use (for most classifiers)

randforclf = False ##use the random forest classifier in this run?
no_of_trees = 240 #number of trees in the forest of the random forest classifier

adaboost = True #use adaboost in this run?
no_est_ada = 100 #number of weak estimators used for adaboost


### Load & Filter dataset
print("Load Dataset & Filter")
df = pd.read_csv("labeled_data.csv")
df_selected = df.drop(['Unnamed: 0','count', "offensive_language", "neither", 'class'], axis=1)


### convert remaining data to machine learning readable form
print("Convert to Array")
hate_speech = np.asarray(df_selected.hate_speech)
tweets = np.asarray(df_selected.tweet)

print("Countvec")
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(tweets)

print("TfidfTrans")
tfidf_transformer = TfidfTransformer()
tweets_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print (tweets_tfidf.shape)


### split the data
print("Split the Data")
tweets_tfidf_train, tweets_tfidf_test, hate_speech_train, hate_speech_test = train_test_split(
    tweets_tfidf, hate_speech, 
    test_size=0.20, random_state=42)


### train the classifier using the training data


if randforclf == True:

    print(f"Training Random Forest Classifier with {no_of_trees} estimators")
    start = time.time()
    clf = RandomForestClassifier(n_estimators=no_of_trees, n_jobs=cores)
    clf.fit(tweets_tfidf_train, hate_speech_train)

    ### compute accuracy using test data
    acc_test = clf.score(tweets_tfidf_test, hate_speech_test)
    print ("Test Accuracy:", acc_test)
    end = time.time()
    print ("Training Time: ", end - start)


if adaboost == True:
    start = time.time()
    print(f"training AdaBoost with {no_est_ada} estimators")
    clf = AdaBoostClassifier(n_estimators=no_est_ada)  # , n_jobs=cores)
    clf.fit(tweets_tfidf_train, hate_speech_train)

    ### compute accuracy using test data
    acc_test = clf.score(tweets_tfidf_test, hate_speech_test)
    print("Test Accuracy:", acc_test)
    end = time.time()
    print ("Training Time: ", end - start)
