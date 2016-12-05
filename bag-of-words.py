import csv
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from processor import processData
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt



vectorizer = CountVectorizer(binary=True, stop_words='english')
# vectorizer = TfidfVectorizer(min_df=1)



def create_tfidf_training_data(document):

    # Create the training data class labels
    y = [d[1] for d in document]

    # Create the document corpus list
    corpus = [d[0] for d in document]

    # Create the TF-IDF vectoriser and transform the corpus

    X = vectorizer.fit_transform(corpus)
    return X, y

#train svm classifier
def train_svm(X, y):

    svm = SVC(C=100, gamma=0.0005, kernel='rbf')
    svm.fit(X, y)
    return svm

#train nb
def train_BNB(X, y):
    bnb = BernoulliNB()
    bnb.fit(X_train, y_train)
    BernoulliNB(alpha=0.5, binarize=1, class_prior=None, fit_prior=False)

    return bnb

def most_informative_feature_for_binary_classification(vectorizer, classifier, n=20):
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
    with open('change-500.csv', 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            # arrange file content in the tuple, push to documents array
            allContent.append(row[2])
            allCodes.append(row[6])

    allContent = processData(allContent)

    docs = list(zip(allContent, allCodes))

# Vectorize and TF-IDF transform the corpus
    X, y = create_tfidf_training_data(docs)


    # Create the training-test split of the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    #Create and train the Support Vector Machine
    svm = train_svm(X_train, y_train)
    pred = svm.predict(X_test)
    print "SVM Result:"
    print svm.score(X_test, y_test)
    print confusion_matrix(pred, y_test)
    print "true positive:"
    print precision_score(y_test, pred, labels=None, pos_label="0", average='binary', sample_weight=None)
    print "F measurement:"
    print f1_score(y_test, pred, average='macro')
    print f1_score(y_test, pred, average='micro')
    print f1_score(y_test, pred, average='weighted')




    # bnb = train_BNB(X_train, y_train)
    # print("nb Result:")
    # print(bnb.score(X_test, y_test))

    # most_informative_feature_for_binary_classification(vectorizer, svm)



# from sklearn.model_selection import ParameterGrid
# param_grid = {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
# list(ParameterGrid(param_grid))
#
# [{'C': 1, 'gamma': 0.001},
# {'C': 1, 'gamma': 0.0001},
# {'C': 10, 'gamma': 0.001},
# {'C': 10, 'gamma': 0.0001},
# {'C': 100, 'gamma': 0.001},
# {'C': 100, 'gamma': 0.0001},
# {'C': 1000, 'gamma': 0.001},
# {'C': 1000, 'gamma': 0.0001}]
#
# # grid_search.py
# import datetime
# import sklearn
# from sklearn import cross_validation
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report
# from sklearn.svm import SVC
#
# if __name__ == "__main__":
#     docs = []
#     allContent = []
#     allCodes = []
#
#     # open the coded tweets csv file
#     with open('test-01.csv', 'rb') as f:
#         reader = csv.reader(f, delimiter=',')
#         for row in reader:
#             # arrange file content in the tuple, push to documents array
#             allContent.append(row[2])
#             allCodes.append(row[4])
#
#     allContent = processData(allContent)
#
#     docs = list(zip(allContent, allCodes))
#
#     # Train/test split
#     X, y = create_tfidf_training_data(docs)
#     X_train, X_test, y_train, y_test = train_test_split(
#                 X, y, test_size=0.3, random_state=42
#             )
#
#     # Set the parameters by cross-validation
#     tuned_parameters = [
#         {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}
#     ]
#     # Perform the grid search on the tuned parameters
#     model = GridSearchCV(SVC(C=1), tuned_parameters, cv=10)
#     model.fit(X_train, y_train)
#     print "Optimised parameters found on training set:"
#     print model.best_estimator_, "\n"
#     print "Grid scores calculated on training set:"
#     for params, mean_score, scores in model.grid_scores_:
#         print "%0.3f for %r" % (mean_score, params)



