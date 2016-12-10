# -*- coding: utf-8 -*-

import sys
import re
import csv
import string
import nltk
from nltk.corpus import stopwords

from sklearn.preprocessing import LabelEncoder
from url_classifier import getURLType
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import *


featureSets = []

def countNumberOfWords(tweet):
    return len(tweet.split())

def countChars(tweet):
    return len(tweet) - tweet.count(' ')

def countNumericChars(tweet):
    return sum(c.isdigit() for c in tweet)


def countHashtag(tweet):
    hashTagCounter = 0
    for word in tweet.split():
        if word.startswith("#"):
            hashTagCounter += 1
    return hashTagCounter

def countAtUser(tweet):
    atUserCounter = 0
    for word in tweet.split():
        if word.startswith("@"):
            atUserCounter += 1
    return atUserCounter

def countUrl(tweet):
    urlCounter = 0
    for word in tweet.split():
        if word.startswith("http"):
            urlCounter += 1
    return urlCounter

def countEmoji(tweet):
    return len(re.findall(u'[\U0001f600-\U0001f650]', tweet))

def countHashtagPerWord(tweet):
    return countHashtag(tweet) / float(len(tweet.split()))

def countUrlPerWord(tweet):
    return countUrl(tweet) / float(len(tweet.split()))

def countEmojiPerWord(tweet):
    return countEmoji(tweet) / float(len(tweet.split()))

def ifStartWithHashtag(tweet):
    return tweet[0] == "#"


def ifEndWithHashtag(tweet):
    return tweet.split()[-1][0] == "#"

def ifStartWithUrl(tweet):
    return tweet.startswith("http")

def ifEndWithUrl(tweet):
    return tweet.split()[-1].startswith("http")

def replaceAtUser(tweet):
    tweet = re.sub(r'[@]\S*', '@user', tweet)

    return tweet

def ifHasDealFeature(tweet):
    dealSigns = ["$", "free", "check out", "deal", "%", "watch now", "now live"]

    for word in dealSigns:
        if tweet.find(word) != -1:
            return True

def removeHashtag(tweet):
    tweet = re.sub(r'[#]', '', tweet)

    return tweet

def classifyURL(tweet):
    urlTypes = ""
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet)
    for url in urls:
        urlTypes += getURLType(url)

    return urlTypes


def cleanUrl(tweet):
    tweet = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'url', tweet)

    return tweet

def cleanRepeat(tweet):
    tweet = re.sub(r'(.)\1+', r'\1\1', tweet)

    return tweet

def processData(allTweets):

    allUrls = []
    urls = open("urls.txt", "r")
    for line in urls:
        allUrls.append(line[:-1])


    for i, tweet in enumerate(allTweets):

        features = {}

        features["urlType"] = allUrls[i]

        if ifStartWithHashtag(tweet):
            features["startWithHashtag"] = True
        elif ifEndWithHashtag(tweet):
            features["endWithHashTag"] = True

        if ifStartWithUrl(tweet):
            features["startWithUrl"] = True
        elif ifEndWithUrl(tweet):
            features["endWithUrl"] = True

        features["atUserCount"] = countAtUser(tweet)

        features["urlCount"] = countUrl(tweet)

        # features["urlType"] = classifyURL(tweet)

        features["hashtagCount"] = countHashtag(tweet)

        features["hashtagPerWord"] = countHashtagPerWord(tweet)

        features["urlPerWord"] = countUrlPerWord(tweet)

        features["emojiCount"] = countEmoji(tweet)

        features["emojiPerWord"] = countEmojiPerWord(tweet)

        features["numberOfWords"] = countNumberOfWords(tweet)

        features["numberOfNumericChars"] = countNumericChars(tweet)

        features["numberOfChars"] = countChars(tweet)

        if ifHasDealFeature(tweet):
            features["dealLike"] = True

        tweet = cleanUrl(tweet)

        tweet = replaceAtUser(tweet)

        tweet = removeHashtag(tweet)

        tweet = tweet.lower()

        tweet = cleanRepeat(tweet)

        tknzr = TweetTokenizer()

        terms = tknzr.tokenize(tweet)

        stemmer = PorterStemmer()

        tweetStems = [stemmer.stem(word) for word in terms]

        stop = stopwords.words('english')

        cleanWords = [w for w in tweetStems if w not in stop]

        for t in cleanWords:
            if t not in features:
                features[t] = 1

        featureSets.append(features)

        print("finished")
        print(i)

    return featureSets

#!/usr/bin/python

#  pre-load url classfications result to the file urls.txt
#
# # urls = open("urls.txt", "wb")
# #
# # for i, tweet in enumerate(allContent):
# #     urls.write(classifyURL(tweet) + "\n")
# #     print "finished"
# #     print i
# # # Close opend file97
# # urls.close()
# #
# # file = open("urls.txt", "rb")
# #
# # allUrls = []
# #
# # for line in file:
# #     allUrls.append(line[:-1])
# #
# # print allUrls
#

# testdata = ["'@remy: This is waaaaayyyy too much for you!!!!!!'"]
#
# print(cleanRepeat(testdata[0]))