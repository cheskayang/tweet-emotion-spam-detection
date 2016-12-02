import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np



def create_tfidf_training_data(docs):

    # Create the training data class labels
    y = [d[1] for d in docs]

    # Create the document corpus list
    corpus = [d[0] for d in docs]

    # Create the TF-IDF vectoriser and transform the corpus
    # vectorizer = TfidfVectorizer(min_df=1)
    # vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
    #                              stop_words='english')
    vectorizer = CountVectorizer(min_df=1, binary=True)

    X = vectorizer.fit_transform(corpus).todense
    return X, y

# def train_svm(X, y):
#
#     svm = SVC(C=1000000.0, gamma=0.5, kernel='rbf')
#     svm.fit(X, y)
#     return svm

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()


def train_naive(X, y):
    y_pred = gnb.fit(X, y).predict(X)
    print("Number of mislabeled points out of a total %d points : %d"
        % (X.shape[0],(y != y_pred).sum()))


if __name__ == "__main__":
    docs = []

    # open the coded tweets csv file
    with open('tweets-coded-all.csv', 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            # arrange file content in the tuple, push to documents array
            docs.append((row[0], row[1]))

    # Vectorize and TF-IDF transform the corpus
    X, y = create_tfidf_training_data(docs)

    # Create the training-test split of the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Create and train the Support Vector Machine
    # svm = train_svm(X_train, y_train)

    # Make an array of predictions on the test set
    # pred = svm.predict(X_test)

    # Output the hit-rate and the confusion matrix for each model
    # print(svm.score(X_test, y_test))
    # print(confusion_matrix(pred, y_test))
    train_naive(X_train, y_train)

