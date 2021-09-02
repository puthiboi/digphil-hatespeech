import pandas as pd
import numpy as np
import csv
import re ##regular expressions
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

###Set up the classifier 
cores = 3 # how many cpu cores one wants to use (for most classifiers)
print_to_csv = True ## if the detected tweets should be printed out to .csv
printTweets = False ## if the detected tweets should be printed out to the console too

###CHOOSE AN ALGORITHM###
randforclf = True ##use the random forest classifier in this run?
no_of_trees = 240 #number of trees in the forest of the random forest classifier
linsvc = False ## use linearSVC
mnb = False ##multinomial naive bayes

##set up the other pipeline steps
selectkbest=10000 ## specify the k of the kbest features to select
testSplitSize = 0.2 ##what percentage the test data has in the split

ownPredict = "queer myself" ## give the classifier a test string to classify


#######  Defining the function to predict both hatespeech and  offensive language that can print to console or .csv   #########################################

def hate_offlang_predict(X_train,y_train,soc,song,hate_or_offlang):
##hate or offlang is a string. Either "hate_speech" or "offensive_language" are allowed (actually i)
##soc and song are the paths to the scraped twitter csvs of soccer and eurovision respectively
##algorithm: mnb = multinomial naive bayes, lsvc = linearsvc, rfc=random forest classifier
    
##splitting into train and test data
    X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size=testSplitSize)

#fitting the model
    print("fitting " + hate_or_offlang)
    model = pipeline.fit(X_train,y_train)

    print("Acc-Score: " + str(model.score(X_test,y_test)))
    print("Test prediction of:" + ownPredict + str(model.predict([ownPredict])))
    

    print("predicting soc")
    predictionsoc = model.predict(soc) ##predicting based on the cleaned tweets but outputting the uncleaned ones below
    detectedSoc = []
    print("Hatespeech detected:")
    countSoc = 0
    for item in zip(predictionsoc,df_soc_selected.Text): ##zipping the uncleaned ones with the prediction
        if item[0] != 0:
            if printTweets == True:
                print(item[1])
            countSoc += 1
            detectedSoc.append(item)
    print(f"Number of European championship tweets incorporating {hate_or_offlang} detected: {countSoc}")
    
    if print_to_csv == True:
        with open('detectedsoc '+ hate_or_offlang+'.csv','wt') as out:
            csv_out=csv.writer(out)
            csv_out.writerows(detectedSoc)
        
    print("predicting song")
    prediction = model.predict(df_song_cleaned)
    detectedSong = []
    countSong = 0
    for item in zip(prediction,df_song_selected.Text): 
        if item[0] != 0:
            if printTweets == True:
                print(item[1])
            countSong += 1
            detectedSong.append(item)
    print(f"Number of Eurovision tweets incorporating {hate_or_offlang} detected: {countSong}")

    if print_to_csv == True:
        with open(('detectedsong ' + hate_or_offlang + '.csv'),'wt') as out:
            csv_out=csv.writer(out)
            csv_out.writerows(detectedSong)
    
    return predictionsoc, prediction

############################################################################################


### Load, clean & Filter dataset
print("Load Training Dataset, Clean & Filter")

def clean(dataframe):
    stop = list(stopwords.words("english"))
    #for i,item in enumerate(stop):
      #  if item == "not" or 'no':
        #    del stop[i]
    #print(stop)
    stemmer = SnowballStemmer("english")
    #dataframe = dataframe.apply(lambda x: " ".join([for i in (re.sub("[^a-zA-Z]", ' ',x).split()) if i not in stop]).lower())
    dataframe = dataframe.apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in stop]).lower())
    return dataframe


df = pd.read_csv("../input/hate-speech-and-offensive-language-dataset/labeled_data.csv")
df_selected = df.drop(['Unnamed: 0','count', "neither", 'class'], axis=1)
df_selected = df_selected.fillna(' ')
train_test_cleaned = df_selected.tweet # clean(df_selected.tweet)

print("Loading prediction data")
###eurosoc
df_soc = pd.read_csv('../input/eurosocsong/eurosoc.csv',
                    lineterminator='\n')
df_soc_selected = df_soc.drop(['Datetime','Username'], axis=1)
df_soc_selected = df_soc_selected.fillna(' ')
print("cleaning soc")
df_soc_cleaned= df_soc_selected.Text #np.asarray(clean(df_soc_selected.Text))

###eurosong
df_song = pd.read_csv('../input/eurosocsong/eurosong.csv',
                    lineterminator='\n')
df_song_selected = df_song.drop(['Datetime','Username'], axis=1)
print("cleaning song")
df_song_cleaned= df_song_selected.Text #np.asarray(clean(df_song_selected.Text))


if randforclf == True:
    pipeline = Pipeline([('vect',TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)),
                         ('chi',SelectKBest(chi2,k=selectkbest)),
                     ('clf',RandomForestClassifier(n_estimators = no_of_trees,n_jobs=cores))])
    print(f"Training Random Forest Classifier with {no_of_trees} estimators")
elif linsvc == True:
    pipeline = Pipeline([('vect',TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)),
                         ('chi',SelectKBest(chi2,k=selectkbest)),
                     ('clf',LinearSVC(C=1.0,penalty="l1",max_iter=3000,dual=False))])
    print(f"Training LinearSVC")
elif mnb == True:
    pipeline = Pipeline([('vect',CountVectorizer(ngram_range=(1,2))),('clf',MultinomialNB())])
    print(f"Training multinomial Naive Bayes classifier")


############     Actual Prediction    ###############
pred_soc_hate, pred_song_hate = hate_offlang_predict(train_tweet_cleaned,df_selected.hate_speech,df_soc_cleaned,df_song_cleaned,"hate_speech")

#pred_soc_offlang, pred_song_offlang = hate_offlang_predict(train_tweet_cleaned,df_selected.offensive_language,df_soc_cleaned,df_song_cleaned,"offensive_language")
