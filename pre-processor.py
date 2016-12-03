import re

testData = ["CHECK IT OUT http://ebay.to/1E668IW #Anger Inside Out Small #Figure #sales #InsideOut #ebay #shopping",
            "@realDonaldTrump good you racist pig #angry as hell Blackman",
            "IS ME!! @djfitix #Play #music #MAD TOUR",
            "#Angry #Birds #Stella - #Season 2 Ep.5 #Sneak #Peek - ... http://wp.me/p5wiVg-dnb #Ep5 #Its #Minequot"]


def countAndAddHashtagFeature(tweet):
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

def processData(allTweets):

    cleanTweets = []

    for tweet in allTweets:

        if ifStartWithHashtag(tweet):
            tweet += " f-start-hashtag"
        if ifEndWithHashtag(tweet):
            tweet += " f-end-hashtag"

        counter = countAndAddHashtagFeature(tweet)
        if counter == 0:
            tweet += " f-0-hashtag"

        elif counter > 0 and counter <= 3:
            tweet += " f-03-hashtag"

        elif counter > 3 and counter <= 5:
            tweet += " f-35-hashtag"

        elif counter > 5 and counter <= 7:
            tweet += " f-57-hashtag"

        elif counter > 7:
            tweet += " f-8-hashtag"

        tweet = replaceAtUser(tweet)

        tweet = removeHashtag(tweet)

        tweet = tweet.lower()

        cleanTweets.append(tweet)

    return cleanTweets


print processData(testData)
