import snscrape.modules.twitter as sntwitter
import pandas as pd

# Creating list to append tweet data to
tweets_list1 = []
tweets_list2 = []

###European Soccer Championship 2021
# Using TwitterSearchScraper to scrape data and append tweets to list
for i,tweet in enumerate(sntwitter.TwitterSearchScraper('#euro2021 lang:en since:2021-06-11 until:2021-07-12').get_items()):
    if i>10000000:
        break
    tweets_list1.append([tweet.date, tweet.content, tweet.user.username])
    
# Creating a dataframe from the tweets list above
tweets_df = pd.DataFrame(tweets_list1, columns=['Datetime', 'Text', 'Username'])

#load data into a .csv file
tweets_df.to_csv ('eurosoc.csv', index = False, header=True)


###Eurovision Song Contest 2021
# Using TwitterSearchScraper to scrape data and append tweets to list
for i,tweet in enumerate(sntwitter.TwitterSearchScraper('#eurovision lang:en since:2021-05-18 until:2021-05-23').get_items()):
    if i>10000000:
        break
    tweets_list2.append([tweet.date, tweet.content, tweet.user.username])
    
# Creating a dataframe from the tweets list above
tweets_df2 = pd.DataFrame(tweets_list2, columns=['Datetime', 'Text', 'Username'])

#load data into a .csv file
tweets_df2.to_csv ('eurosong.csv', index = False, header=True)