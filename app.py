# -*- coding: utf-8 -*-

import re
import urllib2
import codecs

# Use to unwrap a shortened URL
class HeadRequest(urllib2.Request):
    def get_method(self):
        return "HEAD"

# Check if any external url exists in a tweet
def hasExternalURL(tweet):
    bFound = False;
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet)
    for url in urls:
        if url.find('https://t.co/') != -1:
            try:
                res = urllib2.urlopen(HeadRequest(url))
                bFound = (res.geturl().find('twitter.com') == -1)
            except:
                bFound = True
        else:
            bFound = (url.find('twitter.com') == -1)
            
    return bFound

#count the number of hashtags in a tweet and return the number#
def getNumberOfHashtags(tweet):
    counter = 0
    for word in tweet.split():
        if word.startswith("#"):
            counter += 1

    return counter

def checkSpam(tweet):
    score = 0.00
    numOfHashtags = float(getNumberOfHashtags(tweet))
    bHasExternalURL = hasExternalURL(tweet)
    # print bHasExternalURL, numOfHashtags
    if bHasExternalURL:
        if numOfHashtags > 0:
            score = 0.65 + ((numOfHashtags - 1) / numOfHashtags * 0.35)
        else:
            score = 0.60
    else:
        if numOfHashtags > 2:
            score = 0.40 + ((numOfHashtags - 3) / numOfHashtags * 0.60)
            
    return score #(score > 0.60)

infile = codecs.open("sample.csv", "r", "utf_8")
tweets_raw = infile.readlines()
for tweet in tweets_raw:
    tweet = tweet[:-4]
    print checkSpam(tweet)
    
# tweets_data = csv.reader(infile)
# for idx, row in enumerate(tweets_data):
#     print idx

# print checkSpam("start being excited about what could go right. #life #happy #quotes #inspiration #motivation #love #win #sad #quote https://t.co/yUheMRi1yr")