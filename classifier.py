import pandas as pd
import numpy as np
import time
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


###Set up the classifier 

cores = 2 # how many cpu cores one wants to use (for most classifiers)

randforclf = True #use random forest classifier in this run?
no_of_trees = 240 #num of trees in random forest classifier

adaboost = False #use adaboost in this run?
no_est_ada = 100 #num of weak estimators used for adaboost


###Prepare Training Data

def prep_train():
    ### Load & Filter dataset
    print("Load Dataset & Filter")
    df = pd.read_csv("labeled_data.csv")
    
    df_selected_hate_speech = df.drop(['Unnamed: 0','count', "offensive_language", "neither", 'class'], axis=1)
    df_selected_hate_speech = df_selected_hate_speech.fillna(' ')
    
    df_selected_off_lang = df.drop(['Unnamed: 0','count', "hate_speech", "neither", 'class'], axis=1)
    df_selected_off_lang = df_selected_off_lang.fillna(' ')
    

    ### convert remaining data to machine learning readable form
    print("Convert to Array")
    hate_speech = np.asarray(df_selected_hate_speech.hate_speech)
    off_lang = np.asarray(df_selected_off_lang.offensive_language)
    tweets = np.asarray(df_selected_hate_speech.tweet)
    train_len = len(tweets)
    return hate_speech, off_lang, tweets, train_len


hate_speech, off_lang, tweets, train_len = prep_train()


###Prepare Prediction Data

def prep_predict ():
    ### Load & Filter dataset
    print("Load Dataset & Filter")

    ###eurosoc
    df_soc = pd.read_csv('eurosoc.csv',
                    lineterminator='\n')
    df_soc_selected = df_soc.drop(['Datetime','Username'], axis=1)
    df_soc_selected = df_soc_selected.fillna(' ')

    ###eurosong
    df_song = pd.read_csv('eurosong.csv',
                    lineterminator='\n')
    df_song_selected = df_song.drop(['Datetime','Username'], axis=1)


    ### convert remaining data to machine learning readable form
    print("Convert to Array")
    text_soc = np.asarray(df_soc_selected.Text)
    text_song = np.asarray(df_song_selected.Text)
    return text_soc, text_song


text_soc, text_song = prep_predict()


###combine training data and prediction data

combined_soc = np.concatenate((tweets, text_soc))
combined_song = np.concatenate((tweets, text_song))

combined_len_soc = len(combined_soc)
combined_len_song = len(combined_song)

###Vectorize Combined Data

print("Countvec")
count_vect_soc = CountVectorizer()
count_vect_song = CountVectorizer()

X_train_counts_soc = count_vect_soc.fit_transform(combined_soc)
X_train_counts_song = count_vect_song.fit_transform(combined_song)


print("TfidfTrans")
tfidf_transformer_soc = TfidfTransformer()
tfidf_transformer_song = TfidfTransformer()

tweets_tfidf_soc = tfidf_transformer_soc.fit_transform(X_train_counts_soc)
tweets_tfidf_song = tfidf_transformer_song.fit_transform(X_train_counts_song)

###split training data and prediction data
train_soc = tweets_tfidf_soc[0:train_len]
soc = tweets_tfidf_soc[train_len:]

train_song = tweets_tfidf_song[0:train_len]
song = tweets_tfidf_song[train_len:]

###check if split was succesful
print("Train_len: ", train_len)
print("Train: ", train_soc.shape)
print("Soc_len: ", combined_len_soc - train_len)
print("Soc: ", soc.shape)


### train the classifier using training data

def classifier(event ,label):
    if randforclf == True:

        print(f"Training Random Forest Classifier with {no_of_trees} estimators")
        start = time.time()
        clf = RandomForestClassifier(n_estimators=no_of_trees, n_jobs=cores)
        clf.fit(event, label)

        end = time.time()
        print ("Training Time: ", end - start)
        

    if adaboost == True:
        start = time.time()
        print(f"training AdaBoost with {no_est_ada} estimators")
        clf = AdaBoostClassifier(n_estimators=no_est_ada)  # , n_jobs=cores)
        clf.fit(event, label)

        end = time.time()
        print ("Training Time: ", end - start)
    return clf


clf_hate_speech_soc = classifier(train_soc, hate_speech)
clf_off_lang_soc = classifier(train_soc, off_lang)

clf_hate_speech_song = classifier(train_song, hate_speech)
clf_off_lang_song = classifier(train_song, off_lang)

    
### Prediction

start2 = time.time()    

hate_speech_new_soc = clf_hate_speech_soc.predict(soc)
off_lang_new_soc = clf_off_lang_soc.predict(soc)

hate_speech_new_song = clf_hate_speech_song.predict(song)
off_lang_new_song = clf_off_lang_song.predict(song)
#print("Prediction_len: ", len(hate_speech_new_soc))

end2 = time.time()
print ("Prediction Time: ", end2 - start2)


###Save Prediction to CSV

l1 = list(zip(hate_speech_new_soc, off_lang_new_soc, text_soc))
l2 = list(zip(hate_speech_new_song, off_lang_new_song, text_song))

with open('soc.csv','wt') as out:
        csv_out=csv.writer(out)
        csv_out.writerows(l1)

with open('song.csv','wt') as out:
        csv_out=csv.writer(out)
        csv_out.writerows(l2)
