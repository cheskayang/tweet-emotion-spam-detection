# -*- coding: utf-8 -*-

import re
import csv
from url_classifier import getURLType

testData = ["CHECK IT OUT http://ebay.to/1E668IW #Anger Inside Out Small #Figure #sales #InsideOut #ebay #shopping",
            "@realDonaldTrump good you racist pig #angry as hell @Blackman",
            "IS ME!! @djfitix #Play #music #MAD TOUR",
            "#Angry #Birds #Stella - #Season 2 Ep.5 #Sneak #Peek - ... http://wp.me/p5wiVg-dnb #Ep5 #Its #Minequot"]

featureSets = []

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

def countHashtagPerWord(tweet):
    return len(tweet.split()) / countHashtag(tweet)


def ifStartWithHashtag(tweet):
    return tweet[0] == "#"


def ifEndWithHashtag(tweet):
    return tweet.split()[-1][0] == "#"

def ifStartWithUrl(tweet):
    return tweet[0] == "http"


def ifEndWithUrl(tweet):
    return tweet.split()[-1][0] == "http"

def replaceAtUser(tweet):
    tweet = re.sub(r'[@]\S*', '@user', tweet)

    return tweet

def removeHashtag(tweet):
    tweet = re.sub(r'[#]', '', tweet)

    return tweet

def classifyURL(tweet):
    urlTypes = ""
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet)
    for url in urls:
        urlTypes += getURLType(url)

    return urlTypes


def ifHasDealFeature(tweet):
    # dealSigns = ["$", "free", "check out", "deal", "%"]
    #
    #
    if tweet.find("$") != -1 or tweet.find("free") != -1:
        return True

def processData(allTweets):

    for i, tweet in enumerate(allTweets):

        features = {}

        if ifStartWithHashtag(tweet):
            features["hashtagPosition"] = "start"
        elif ifEndWithHashtag(tweet):
            features["hashtagPosition"] = "end"

        if ifStartWithUrl(tweet):
            features["urlPosition"] = "start"
        elif ifEndWithUrl(tweet):
            features["urlPosition"] = "end"

        features["atUserCount"] = countAtUser(tweet)

        features["urlCount"] = countUrl(tweet)

        # features["urlType"] = classifyURL(tweet)

        features["hashtagCount"] = countHashtag(tweet)

        features["hashtagPerWord"] = countHashtagPerWord(tweet)

        if ifHasDealFeature(tweet):
            features["dealLike"] = True

        featureSets.append(features)

        print "finished"
        print i

    return featureSets


docs = []
allContent = []
allCodes = []

# open the coded tweets csv file
with open('change-500.csv', 'rb') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        # arrange file content in the tuple, push to documents array
        allContent.append(row[2])
        allCodes.append(row[6])

allContent = processData(allContent)

docs = list(zip(allContent, allCodes))

from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB



vectorizer = DictVectorizer()

def train_svm(X, y):

    svm = SVC(C=1000, gamma=0.001, kernel='rbf')
    svm.fit(X, y)
    return svm

def train_BNB(X, y):
    bnb = BernoulliNB()
    bnb.fit(X_train, y_train)
    return bnb

def create_tfidf_training_data(document):

    # Create the training data class labels
    y = [d[1] for d in document]

    # Create the document corpus list
    corpus = [d[0] for d in document]

    # Create the TF-IDF vectoriser and transform the corpus

    X = vectorizer.fit_transform(corpus)
    return X, y

X, y = create_tfidf_training_data(docs)


# Create the training-test split of the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Create and train the Support Vector Machine
svm = train_svm(X_train, y_train)
print("SVM Result:")
print(svm.score(X_test, y_test))


# #naive bayes
# def most_informative_feature_for_binary_classification(vectorizer, classifier, n=100):
#     class_labels = classifier.classes_
#     feature_names = vectorizer.get_feature_names()
#     topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
#     topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]
#
#     for coef, feat in topn_class1:
#         print class_labels[0], coef, feat
#
#     print
#
#     for coef, feat in reversed(topn_class2):
#         print class_labels[1], coef, feat
#
# bnb = train_BNB(X_train, y_train)
# print("nb Result:")
# print(bnb.score(X_test, y_test))

# most_informative_feature_for_binary_classification(vectorizer, bnb)
