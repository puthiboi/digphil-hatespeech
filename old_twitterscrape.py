import sys
import tweepy
import csv
import pandas as pd
import time


try:
    from credentials import consumer_key, consumer_secret, access_token, access_token_secret
except ModuleNotFoundError:
    sys.exit('credentials.py does not exist in src/ folder. '
             'See README.md for instructions.')
except ImportError as import_error:
    sys.exit(f'{import_error}\nCheck for spelling.')

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

#5 minutes from now to break for loop
timeout = time.time() + 60*5

#####EuroVisionSongContest
# Open/Create a file to append data
csvFile = open('eurosong2021.csv', 'a')
#Use csv Writer
csvWriter = csv.writer(csvFile)

for tweet in tweepy.Cursor(api.search,q="eurovision",count=100,
                           lang="en",
                           since="2021-06-11"
                           ).items():
    print (tweet.created_at, tweet.text)
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
    if time.time() > timeout:
        break




    