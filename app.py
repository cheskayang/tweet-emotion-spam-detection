# -*- coding: utf-8 -*-

import re
import urllib2
import codecs
import csv

whiteList = ["twitter.com", "instagram.com", "telegraph.co.uk", "sky.com", "apple.news", "statnews.com"]
shortList = ["//t.co/", "//ow.ly/", "//goo.gl/", "//lnkd.in/", "//ift.tt/"]

# Use to unwrap a shortened URL
class HeadRequest(urllib2.Request):
    def get_method(self):
        return "HEAD"

def isExemption(url):
    if url.find('/news/') == -1 and url.find('/articles/') == -1:
        for white in whiteList:
            if url.find(white) != -1:
                return True
        return False
    else:
        return True

def isShortURL(url):
    for short in shortList:
        if url.find(short) != -1:
            return True

# Check if any external url exists in a tweet
def hasSuspectExternalURL(tweet):
    bFound = False;
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet)
    for url in urls:
        if isShortURL(url):
            try:
                res = urllib2.urlopen(HeadRequest(url))
                bFound = not isExemption(res.geturl()) #(res.geturl().find('twitter.com') == -1)
            except:
                bFound = True
        else:
            bFound = not isExemption(url) #(url.find('twitter.com') == -1)
            
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
    bHasExternalURL = hasSuspectExternalURL(tweet)
    # print bHasExternalURL, numOfHashtags
    if bHasExternalURL:
        if numOfHashtags > 0:
            score = 0.65 + ((numOfHashtags - 1) / numOfHashtags * 0.35)
        else:
            score = 0.60
    else:
        if numOfHashtags > 2:
            score = 0.30 + ((numOfHashtags - 3) / numOfHashtags * 0.70)
            
    return score #(score > 0.60)

# Test the algorithm
with open('sample_0.csv', 'rb') as f:
    SPAM_THRESHOLD_SCORE = 0.55
    reader = csv.reader(f, delimiter=',')
    spam_counter = 0
    is_Spam = False
    for i, row in enumerate(reader):
        is_Spam = (checkSpam(row[0]) > SPAM_THRESHOLD_SCORE)
        print str(i) + '\t' + str(is_Spam) + '\t' + row[0]
        if is_Spam:
            spam_counter += 1
    print str(float(spam_counter) / (i + 1) * 100) + '% accuracy'
    
# infile = codecs.open("sample_0.csv", "r", "utf_8")
# tweets_raw = infile.readlines()
# for tweet in tweets_raw:
#     tweet = tweet[:-4]
#     print checkSpam(tweet), tweet