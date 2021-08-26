import pandas as pd
import numpy as np


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








