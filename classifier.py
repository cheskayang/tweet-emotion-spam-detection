import csv

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.model_selection import cross_val_score


from sklearn.naive_bayes import BernoulliNB

from feature_extraction import processData


vectorizer = DictVectorizer()


def train_svm(X, y):
    # svm = SVC(kernel='rbf', C=1.0, gamma=0.1)
    svm = SVC(C=0.1, kernel="linear")
    svm.fit(X, y)
    return svm


def train_BNB(X, y):
    bnb = BernoulliNB()
    bnb.fit(X_train, y_train)
    return bnb


def create_training_data(document):

    # Create the training data class labels
    y = [d[1] for d in document]

    # Create the document corpus list
    corpus = [d[0] for d in document]

    # Create the TF-IDF vectoriser and transform the corpus

    X = vectorizer.fit_transform(corpus)
    return X, y

if __name__ == "__main__":

    # prepare documents
    docs = []
    allContent = []
    allCodes = []

    # open the coded tweets csv file
    with open('samples.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            # arrange file content in the tuple, push to documents array
            allContent.append(row[2])
            allCodes.append(row[6])

    allContent = processData(allContent)

    docs = list(zip(allContent, allCodes))

    # fit training data and split for training and testing
    X, y = create_training_data(docs)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    # Create and train the Support Vector Machine
    svm = train_svm(X_train, y_train)
    prediction = svm.predict(X_test)

    # calculate measurements
    accuracy = accuracy_score(y_test, prediction)
    scores = cross_val_score(svm, X_train, y_train, cv=10)
    precision = precision_score(y_test, prediction, labels=None, pos_label="1", average='binary', sample_weight=None)
    macroF1 = f1_score(y_test, prediction, average='macro')
    microF1 = f1_score(y_test, prediction, average='micro')
    weightedF1 = f1_score(y_test, prediction, average='weighted')

    # print measurements
    print("Accuracy: %0.4f" % accuracy)
    print("Precision: %0.4f" % precision)
    print("Macro F1: %0.4f" % macroF1)
    print("Micro F1: %0.4f" % microF1)
    print("Weighted F1: %0.4f" % weightedF1)
    print("#################report##############")
    print(classification_report(y_test, prediction, digits=4))
    print("#######Cross-validation##############")
    print(scores)
    print("Average accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    print("#######confusion matrix##############")
    print(confusion_matrix(prediction, y_test))

    # grid search for finding parameters
    # from sklearn.pipeline import Pipeline
    # from sklearn.model_selection import GridSearchCV
    # pipe_svc = Pipeline([
    #                      ('clf', SVC(random_state=1))])
    # param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    # param_grid = [{'clf__C': param_range,
    #                'clf__kernel': ['linear']},
    #               {'clf__C': param_range,
    #                'clf__gamma': param_range,
    #                'clf__kernel': ['rbf']}]
    #
    # gs = GridSearchCV(estimator=pipe_svc,
    #                  param_grid=param_grid,
    #                  scoring='accuracy',
    #                  cv=10,
    #                  n_jobs=-1)
    # gs = gs.fit(X_train, y_train)
    # print(gs.best_score_)
    # print(gs.best_params_)