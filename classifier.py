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
cleanTweets = False ## if the base tweets should be cleaned or not

###CHOOSE AN ALGORITHM###
randforclf = True ##use the random forest classifier in this run?
no_of_trees = 240 #number of trees in the forest of the random forest classifier
linsvc = False ## use linearSVC
mnb = False ##multinomial naive bayes

##set up the other pipeline steps
ngramrange = 3
selectkbest=10000 ## specify the k of the kbest features to select
testSplitSize = 0.1 ##what percentage the test data has in the split

ownPredict = "am queer myself" ## give the classifier a test string to classify


#######  Defining the function to predict both hatespeech and  offensive language that can print to console or .csv   #########################################
def clean(dataframe):
    print("cleaning in progress...")
    stop = list(stopwords.words("english"))
    #for i,item in enumerate(stop):
      #  if item == "not" or 'no':
        #    del stop[i]
    #print(stop)
    stemmer = SnowballStemmer("english")
    #dataframe = dataframe.apply(lambda x: " ".join([for i in (re.sub("[^a-zA-Z]", ' ',x).split()) if i not in stop]).lower())
    dataframe = dataframe.apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in stop]).lower())
    return dataframe


def hate_offlang_predict(X_train,y_train,soc,song,hate_or_offlang, labelFunc, cleaning):
##hate or offlang is a string. used to mean something, now it's a label for both the file and the fitting X... message
##soc and song are the scraped twitter csvs of soccer and eurovision respectively
##labelFunc = lambda function or else that spits out true/false and orders the output, so only the hatespeech is printed to the console/csv
##cleaning is a boolean if the tweets should be cleaned on both ends or not per the above function
    
    if cleaning == True:
        X_train = clean(X_train)
        soc = clean(soc)
        song = clean(song)
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
        if labelFunc(item[0])==True:
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
        if labelFunc(item[0])==True:
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


########################## Load & Filter training dataset ############################
print("Load Training Dataset, Clean & Filter")

##new training dataset
dfdynGen = pd.read_csv("../input/dynamically-generated-hate-speech-dataset/2020-12-31-DynamicallyGeneratedHateDataset-entries-v0.1.csv")
#print(dfdynGen)
train_dynGen = dfdynGen.text #clean(dfdynGen.text)
dfdynFunc = (lambda x: True if x == 'hate' else False)

##old training dataset
df = pd.read_csv("../input/hate-speech-and-offensive-language-dataset/labeled_data.csv")
df_selected = df.drop(['Unnamed: 0','count', "neither", 'class'], axis=1)
df_selected = df_selected.fillna(' ')
train_tweet = df_selected.tweet # clean(df_selected.tweet)
df_orig_func = (lambda x: True if x > 0 else False)

######################load, filter prediction/target corpus ############################
print("Loading prediction data")
###eurosoc
df_soc = pd.read_csv('eurosoc.csv',
                    lineterminator='\n')
df_soc_selected = df_soc.drop(['Datetime','Username'], axis=1)
df_soc_selected = df_soc_selected.fillna(' ')
df_soc_text= df_soc_selected.Text #np.asarray(clean(df_soc_selected.Text))

###eurosong
df_song = pd.read_csv('eurosong.csv',
                    lineterminator='\n')
df_song_selected = df_song.drop(['Datetime','Username'], axis=1)
df_song_text= df_song_selected.Text #np.asarray(clean(df_song_selected.Text))

#############choosing algorithm######################
if randforclf == True:
    pipeline = Pipeline([('vect',TfidfVectorizer(ngram_range=(1,ngramrange),sublinear_tf=True)),
                         ('chi',SelectKBest(chi2,k=selectkbest)),
                     ('clf',RandomForestClassifier(n_estimators = no_of_trees,n_jobs=cores))])
    print(f"Training Random Forest Classifier with {no_of_trees} estimators")
elif linsvc == True:
    pipeline = Pipeline([('vect',TfidfVectorizer(ngram_range=(1,ngramrange),sublinear_tf=True)),
                         ('chi',SelectKBest(chi2,k=selectkbest)),
                     ('clf',LinearSVC(C=1.0,penalty="l1",max_iter=3000,dual=False))])
    print(f"Training LinearSVC")
elif mnb == True:
    pipeline = Pipeline([('vect',CountVectorizer(ngram_range=(1,ngramrange))),('clf',MultinomialNB())])
    print(f"Training multinomial Naive Bayes classifier")


############     Actual Prediction    ###############
pred_soc_hate, pred_song_hate = hate_offlang_predict(train_dynGen,dfdynGen.label,df_soc_text,df_song_text,"hs_dynGen", dfdynFunc,cleanTweets)
pred_soc_olddata, pred_song_olddata = hate_offlang_predict(train_tweet,df_selected.hate_speech,df_soc_text,df_song_text,'hs_olddata', df_orig_func,cleanTweets)

#pred_soc_offlang, pred_song_offlang = hate_offlang_predict(train_tweet_cleaned,df_selected.offensive_language,df_soc_cleaned,df_song_cleaned,"offensive_language")
