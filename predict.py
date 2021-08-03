import pandas as pd
import numpy as np
import time
#import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


### Load & Filter dataset
print("Load Dataset & Filter")
df_soc = pd.read_csv('eurosoc.csv',
                 lineterminator='\n')
df_soc_selected = df_soc.drop(['Datetime','Username'], axis=1)



### convert remaining data to machine learning readable form
print("Convert to Array")
text_soc = np.asarray(df_soc_selected.Text)

print("Countvec")
count_vect_soc = CountVectorizer()
X_predict_counts_soc = count_vect_soc.fit_transform(text_soc)

print("TfidfTrans")
tfidf_transformer = TfidfTransformer()
tweets_tfidf_soc = tfidf_transformer.fit_transform(X_predict_counts_soc)
print (tweets_tfidf_soc.shape)
