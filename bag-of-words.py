import csv
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB

#process data
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
            tweet += " fstarthashtag"
        if ifEndWithHashtag(tweet):
            tweet += " fendhashtag"

        counter = countAndAddHashtagFeature(tweet)
        if counter == 0:
            tweet += " f0hashtag"

        elif counter > 0 and counter <= 3:
            tweet += " f03hashtag"

        elif counter > 3 and counter <= 5:
            tweet += " f35hashtag"

        elif counter > 5 and counter <= 7:
            tweet += " f57hashtag"

        elif counter > 7:
            tweet += " f8hashtag"

        tweet = replaceAtUser(tweet)

        tweet = removeHashtag(tweet)

        tweet = tweet.lower()

        cleanTweets.append(tweet)

    return cleanTweets


vectorizer = CountVectorizer(min_df=1, binary=True)


def create_tfidf_training_data(document):

    # Create the training data class labels
    y = [d[1] for d in document]

    # Create the document corpus list
    corpus = [d[0] for d in document]

    # Create the TF-IDF vectoriser and transform the corpus
    # vectorizer = TfidfVectorizer(min_df=1)

    X = vectorizer.fit_transform(corpus)
    return X, y

#train svm classifier
# def train_svm(X, y):
#
#     svm = SVC(C=1000000.0, gamma=0.5, kernel='rbf')
#     svm.fit(X, y)
#     return svm

#train nb

def train_BNB(X, y):
    bnb = BernoulliNB()
    bnb.fit(X_train, y_train)
    BernoulliNB(alpha=0.5, binarize=1, class_prior=None, fit_prior=False)

    return bnb

def most_informative_feature_for_binary_classification(vectorizer, classifier, n=10):
    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names()
    topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
    topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]

    for coef, feat in topn_class1:
        print class_labels[0], coef, feat

    print

    for coef, feat in reversed(topn_class2):
        print class_labels[1], coef, feat


if __name__ == "__main__":
    docs = []
    allContent = []
    allCodes = []

    # open the coded tweets csv file
    with open('test-01.csv', 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            # arrange file content in the tuple, push to documents array
            allContent.append(row[2])
            allCodes.append(row[4])

    allContent = processData(allContent)

    docs = list(zip(allContent, allCodes))

# Vectorize and TF-IDF transform the corpus
    X, y = create_tfidf_training_data(docs)


    # Create the training-test split of the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # #Create and train the Support Vector Machine
    # svm = train_svm(X_train, y_train)
    #
    # #Make an array of predictions on the test set
    # pred = svm.predict(X_test)
    #
    # #Output the hit-rate and the confusion matrix for each model
    # print(svm.score(X_test, y_test))
    # print(confusion_matrix(pred, y_test))
    # train_svm(X_train, y_train)

    bnb = train_BNB(X_train, y_train)
    print(bnb.score(X_test, y_test))

    most_informative_feature_for_binary_classification(vectorizer, bnb)




