import pandas as pd
import numpy as np
import csv
import re ##regular expressions
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

###Set up the classifier 

cores = 3 # how many cpu cores one wants to use (for most classifiers)
printTweets = False ## if the detected tweets should be printed out to the console too. If false only a csv will be created

###CHOOSE AN ALGORITHM###
randforclf = False ##use the random forest classifier in this run?
no_of_trees = 240 #number of trees in the forest of the random forest classifier

adaboost = True #use adaboost in this run?
no_est_ada = 140 #number of weak estimators used for adaboost

linsvc = False ## use linearSVC


### Load, clean & Filter dataset
print("Load Dataset & Filter")
stop = stopwords.words("english")
stemmer = SnowballStemmer("english")

df = pd.read_csv("../input/hate-speech-and-offensive-language-dataset/labeled_data.csv")
df_selected = df.drop(['Unnamed: 0','count', "offensive_language", "neither", 'class'], axis=1)
df_selected = df_selected.fillna(' ')

def clean(dataframe):
    #dataframe = dataframe.apply(lambda x: " ".join([for i in (re.sub("[^a-zA-Z]", ' ',x).split()) if i not in stop]).lower())
    dataframe = dataframe.apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in stop]).lower())
    return dataframe

df_selected.tweet = clean(df_selected.tweet)
##splitting into train and test data
X_train,X_test,y_train,y_test = train_test_split(df_selected.tweet,df_selected.hate_speech,test_size=0.2)

if adaboost == True:
    pipeline = Pipeline([('vect',TfidfVectorizer(ngram_range=(1,2),stop_words="english",sublinear_tf=True)),
                     ('clf',AdaBoostClassifier(n_estimators = no_est_ada))])
    print(f"Training AdaBoost with {no_est_ada} estimators")
elif randforclf == True:
    pipeline = Pipeline([('vect',TfidfVectorizer(ngram_range=(1,2),stop_words="english",sublinear_tf=True)),
                     ('clf',RandomForestClassifier(n_estimators = no_of_trees,n_jobs=cores))])
    print(f"Training Random Forest Classifier with {no_of_trees} estimators")
elif linsvc == True:
    pipeline = Pipeline([('vect',TfidfVectorizer(ngram_range=(1,2),stop_words="english",sublinear_tf=True)),
                     ('clf',LinearSVC(C=1.0,max_iter=3000,dual=False))])
    print(f"Training LinearSVC")

#fitting the model
print("fitting...")
model = pipeline.fit(X_train,y_train)

print("Acc-Score: " + str(model.score(X_test,y_test)))

print("Loading prediction data")
###eurosoc
df_soc = pd.read_csv('../input/eurosocsong/eurosoc.csv',
                    lineterminator='\n')
df_soc_selected = df_soc.drop(['Datetime','Username'], axis=1)
df_soc_selected = df_soc_selected.fillna(' ')
print("cleaning soc")
df_soc_cleaned= np.asarray(clean(df_soc_selected.Text))

###eurosong
df_song = pd.read_csv('../input/eurosocsong/eurosong.csv',
                    lineterminator='\n')
df_song_selected = df_song.drop(['Datetime','Username'], axis=1)
print("cleaning song")
df_song_cleaned= np.asarray(clean(df_song_selected.Text))


print("predicting soc")
predictionsoc = model.predict(df_soc_cleaned)
detectedSoc = []
print("Hatespeech detected:")
countSoc = 0
for item in zip(predictionsoc,df_soc_selected.Text):
    if item[0] == 1:
        if printTweets == True:
            print(item[1])
        countSoc += 1
        detectedSoc.append(item)
print(f"Number of European championship tweets incorporating hatespeech detected: {countSoc}")

with open('detectedsoc.csv','wt') as out:
   csv_out=csv.writer(out)
   csv_out.writerows(detectedSoc)
        
print("predicting song")
prediction = model.predict(df_song_cleaned)
detectedSong = []
countSong = 0
for item in zip(prediction,df_song_selected.Text):
    if item[0] == 1:
        if printTweets == True:
            print(item[1])
        countSong += 1
        detectedSong.append(item)
print(f"Number of Eurovision tweets incorporating hatespeech detected: {countSong}")


with open('detectedsong.csv','wt') as out:
   csv_out=csv.writer(out)
   csv_out.writerows(detectedSong)
