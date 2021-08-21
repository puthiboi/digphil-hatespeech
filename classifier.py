import pandas as pd
import numpy as np
import time
import sys
#import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

###Set up the classifier 

cores = 3 # how many cpu cores one wants to use (for most classifiers)

randforclf = True ##use the random forest classifier in this run?
no_of_trees = 240 #number of trees in the forest of the random forest classifier

adaboost = False #use adaboost in this run?
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

### train the classifier using the training data


if randforclf == True:

    print(f"Training Random Forest Classifier with {no_of_trees} estimators")
    start = time.time()
    clf = RandomForestClassifier(n_estimators=no_of_trees, n_jobs=cores)
    clf.fit(tweets_tfidf, hate_speech)

    end = time.time()
    print ("Training Time: ", end - start)
    


if adaboost == True:
    start = time.time()
    print(f"training AdaBoost with {no_est_ada} estimators")
    clf = AdaBoostClassifier(n_estimators=no_est_ada)  # , n_jobs=cores)
    clf.fit(tweets_tfidf, hate_speech)

    end = time.time()
    print ("Training Time: ", end - start)
    
start2 = time.time()    
try:
    from predict import tweets_tfidf_song, tweets_tfidf_soc
except ModuleNotFoundError:
    sys.exit('predict.py import sysdoes not exist in src/ folder. '
             )
except ImportError as import_error:
    sys.exit(f'{import_error}\nCheck for spelling.')

hate_speech_new_soc = clf.predict(tweets_tfidf_soc)

end2 = time.time()
print ("Prediction Time: ", end2 - start2)

#for i in range(len(tweets_tfidf_soc)):
#    print("X=%s, Predicted=%s" % (tweets_tfidf_soc[i], hate_speech_new_soc[i]))