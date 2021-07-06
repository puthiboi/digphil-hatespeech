import tweepy
import csv
import pandas as pd
import time

####input your credentials here
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

#####EuroSoccerChampionship
# Open/Create a file to append data
csvFile = open('eurosoc2021.csv', 'a')
#Use csv Writer
csvWriter = csv.writer(csvFile)

for tweet in tweepy.Cursor(api.search,q="#euro2021",count=100,
                           lang="en",
                           since="2017-04-03").items():
    print (tweet.created_at, tweet.text)
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])

#5 minutes from now to break for loop
timeout = time.time() + 60*5

#####EuroVisionSongContest
# Open/Create a file to append data
csvFile = open('eurosong2021.csv', 'a')
#Use csv Writer
csvWriter = csv.writer(csvFile)

for tweet in tweepy.Cursor(api.search,q="eurovision",count=200,
                           lang="en",
                           since="2021-06-11",
                           until="2021-05-22").items():
    print (tweet.created_at, tweet.text)
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
    if time.time() > timeout:
        break

