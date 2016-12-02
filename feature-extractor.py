import csv
import nltk
from nltk.tokenize import TweetTokenizer

documents = []
tknzr = TweetTokenizer()

# open the coded tweets csv file
with open('tweets-coded-all.csv', 'rb') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        # arrange file content in the tuple (words tokenized from tweet, code), push to documents array
        documents.append((row[0], row[1]))

def findFeatures(document):
    features = {}
    singleTweetWords = []
    singleTweetWords += tknzr.tokenize(document)
    counter = 0
    for word in singleTweetWords:
        if word.startswith("#"):
            counter += 1
    if counter < 2:
        features["numberOfHashtag"] = "<2"
    elif counter == 2:
        features["numberOfHashtag"] = "2"
    elif counter == 3:
        features["numberOfHashtag"] = "3"
    elif counter == 4:
        features["numberOfHashtag"] = "4"
    elif counter == 5:
        features["numberOfHashtag"] = "5"
    elif counter > 5:
        features["numberOfHashtag"] = ">5"

    return features

featuresets = [(findFeatures(doc), code) for (doc, code) in documents]

# print featuresets[0]

# # featuresets = [(findFeatures("#happy #anger had a very bad day!!"), "0")]
#
# print featuresets

# assign part featuresets to training_set and part to test_set
training_set = featuresets[:1000]
testing_set = featuresets[1000:]

# use training_set to train a naive Bayes classifier.
classifier = nltk.NaiveBayesClassifier.train(training_set)

# print classifier accuracy
print("Classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)

# show top 100 features
classifier.show_most_informative_features(5)


