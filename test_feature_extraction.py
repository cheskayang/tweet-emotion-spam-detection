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

def classifyURL(tweet):
    urlTypes = ""
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet)
    for url in urls:
        urlTypes += getURLType(url)

    return urlTypes

def processData(allTweets):

    cleanTweets = []

    for i, tweet in enumerate(allTweets):

        features = {}

        if ifStartWithHashtag(tweet):
            features["hashtagPosition"] = "start"
            if ifEndWithHashtag(tweet):
                features["hashtagPosition"] = "startAndEnd"
        elif ifEndWithHashtag(tweet):
            features["hashtagPosition"] = "end"
        else:
            features["hashtagPosition"] = "end"

        features["atUserCount"] = countAtUser(tweet)

        features["urlType"] = classifyURL(tweet)

        counter = countHashtag(tweet)

        if counter == 0:
            features["hashtagCount"] = "f0hashtag"

        elif counter > 0 and counter <= 3:
            features["hashtagCount"] = " f03hashtag"

        elif counter > 3 and counter <= 5:
            features["hashtagCount"] = " f35hashtag"

        elif counter > 5 and counter <= 7:
            features["hashtagCount"] = " f57hashtag"

        elif counter > 7:
            features["hashtagCount"] = " f8hashtag"

        featureSets.append(features)

        print "finished"
        print i

    return featureSets


docs = []
allContent = []
allCodes = []

# open the coded tweets csv file
with open('test-change-6.csv', 'rb') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        # arrange file content in the tuple, push to documents array
        allContent.append(row[2])
        allCodes.append(row[4])

allContent = processData(allContent)

docs = list(zip(allContent, allCodes))

from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


vectorizer = DictVectorizer()

def train_svm(X, y):

    svm = SVC(C=1000, gamma=0.001, kernel='rbf')
    svm.fit(X, y)
    return svm

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
    X, y, test_size=0.3, random_state=42
)

#Create and train the Support Vector Machine
svm = train_svm(X_train, y_train)
print("SVM Result:")
print(svm.score(X_test, y_test))