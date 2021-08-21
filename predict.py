import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


### Load & Filter dataset
print("Load Dataset & Filter")

###eurosoc
df_soc = pd.read_csv('eurosoc.csv',
                 lineterminator='\n')
df_soc_selected = df_soc.drop(['Datetime','Username'], axis=1)

###eurosong
df_song = pd.read_csv('eurosong.csv',
                 lineterminator='\n')
df_song_selected = df_song.drop(['Datetime','Username'], axis=1)



### convert remaining data to machine learning readable form
print("Convert to Array")
text_soc = np.asarray(df_soc_selected.Text)

text_song = np.asarray(df_song_selected.Text)


print("Countvec")
count_vect_soc = CountVectorizer()
X_predict_counts_soc = count_vect_soc.fit_transform(text_soc)

count_vect_song = CountVectorizer()
X_predict_counts_song = count_vect_song.fit_transform(text_song)

print("TfidfTrans")
tfidf_transformer_soc = TfidfTransformer()
tweets_tfidf_soc = tfidf_transformer_soc.fit_transform(X_predict_counts_soc)

tfidf_transformer_song = TfidfTransformer()
tweets_tfidf_song = tfidf_transformer_song.fit_transform(X_predict_counts_song)

print(tweets_tfidf_soc)
print(tweets_tfidf_song.shape)





