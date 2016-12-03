import re
from url_classifier import getURLType

testData = ["CHECK IT OUT http://ebay.to/1E668IW #Anger Inside Out Small #Figure #sales #InsideOut #ebay #shopping",
            "@realDonaldTrump good you racist pig #angry as hell @Blackman",
            "IS ME!! @djfitix #Play #music #MAD TOUR",
            "#Angry #Birds #Stella - #Season 2 Ep.5 #Sneak #Peek - ... http://wp.me/p5wiVg-dnb #Ep5 #Its #Minequot"]


def countHashtag(tweet):
    hashTagCounter = 0
    for word in tweet.split():
        if word.startswith("#"):
            hashTagCounter += 1
    return hashTagCounter


def ifStartWithHashtag(tweet):
    return tweet[0] == "#"


def ifEndWithHashtag(tweet):
    return tweet.split()[-1][0] == "#"


def replaceAtUser(tweet):
    tweet = re.sub(r'[@]\S*', '@user', tweet)

    return tweet

def removeHashtag(tweet):
    tweet = re.sub(r'[#]', '', tweet)

    return tweet

def replaceUrl(tweet):
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet)
    for url in urls:
        tweet = tweet.replace(url, getURLType(url))
    return tweet

def processData(allTweets):

    featureSet = []
    cleanTweets = []

    for tweet in allTweets:

        features = ''

        if ifStartWithHashtag(tweet):
            features += " fstarthashtag"
        if ifEndWithHashtag(tweet):
            features += " fendhashtag"

        counter = countHashtag(tweet)
        if counter == 0:
            features += " f0hashtag"

        elif counter > 0 and counter <= 3:
            features += " f03hashtag"

        elif counter > 3 and counter <= 5:
            features += " f35hashtag"

        elif counter > 5 and counter <= 7:
            features += " f57hashtag"

        elif counter > 7:
            features += " f8hashtag"

        featureSet.append(features)

        tweet = replaceUrl(tweet)

        tweet = replaceAtUser(tweet)

        tweet = removeHashtag(tweet)

        tweet = tweet.lower()

        tweet += features

        cleanTweets.append(tweet)

    return cleanTweets

#print processData(testData)