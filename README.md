# digphil-hatespeech

# Setup
This app requires Python 3.9 to be installed.

# Usage 
There are two ways to get twitter Datat 'twitterscrape.py' and 'twittersnscrpape.py' with 'twitterscrape' being faster but with less scraped tweets and 'twittersncrape' being easier but slower with unlimited scraped tweets.

# (old) 'twitterscrape.py' Twitter Credentials
You will also need credentials for the Twitter API.
To get them, create a Twitter account and log in to the Twitter Developer Dashboard.
https://developer.twitter.com/en/portal/dashboard
Create a new app. The contents of name and description can be set to arbitrary values.
When you are finished you will see the consumer key, cosumer secret, access token and access token secret.

Create the file `./src/credentials.py` with the following contents:
```
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''
```

# 'twittersncrape.py'
Just run the code.
