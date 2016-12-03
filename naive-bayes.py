import csv
import nltk
from nltk.tokenize import TweetTokenizer

documents = []
allTweetContent = []
tknzr = TweetTokenizer()

# open the coded tweets csv file
with open('tweets-coded-all.csv', 'rb') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        # arrange file content in the tuple (words tokenized from tweet, code), push to documents array
        documents.append((tknzr.tokenize(row[0]), row[1]))
        # push only tweet content into an array
        allTweetContent.append(row[0])

# get all words from tweetContent using tokenizer
allWords = []
for tweet in allTweetContent:
    allWords += tknzr.tokenize(tweet)

# extract top 3000 words and assign to a list
allWords = nltk.FreqDist(allWords)
wordFeatures = list(allWords.keys()[:3000])

# feature extractor which checks whether each of these words is present in a given document.
def findFeatures(document):
    words = set(document)
    features = {}
    for w in wordFeatures:
        features[w] = (w in words)

    return features

featuresets = [(findFeatures(doc), code) for (doc, code) in documents]

print featuresets[0]

# assign part featuresets to training_set and part to test_set
training_set = featuresets[:800]
testing_set = featuresets[800:]

# use training_set to train a naive Bayes classifier.
classifier = nltk.NaiveBayesClassifier.train(training_set)

# print classifier accuracy
print("Classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)

# show top 100 features
classifier.show_most_informative_features(100)


