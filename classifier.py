import pandas as pd
import numpy as np
from numpy import argmax
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


###functional Part

#funtion to convert a listoflists into one big lists 
def flatten(t):
    return [item for sublist in t for item in sublist]

# Load & Filter dataset
df = pd.read_csv("labeled_data.csv")
df_selected = df.drop(['Unnamed: 0','count', "offensive_language", "neither", 'class'], axis=1)

#convert remaining data to array (readable form for machine learning)
hate_speech = np.asarray(df_selected.hate_speech)
tweets = np.asarray(df_selected.tweet)
tweetssplit = [item.split() for item in tweets]
flat_tweetssplit = flatten(tweetssplit)
print(flat_tweetssplit)

###nonfunctional Part

# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(flat_tweetssplit)
#print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
#print(onehot_encoded)
# invert first example
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
#print(inverted)

# split the data
onehot_encoded_train, onehot_encoded_test, hate_speech_train, hate_speech_test = train_test_split(
    onehot_encoded, hate_speech, 
    test_size=0.20, random_state=42)

# initialize
clf = RandomForestClassifier()

# train the classifier using the training data
clf.fit(onehot_encoded_train, hate_speech)

# compute accuracy using test data
acc_test = clf.score(onehot_encoded_test, hate_speech_test)

print ("Test Accuracy:", acc_test)