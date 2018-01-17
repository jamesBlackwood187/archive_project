from __future__ import division
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def accuracy(predictions, actuals):
    return float((np.sign(predictions) == np.sign(actuals)).sum()) / float(len(predictions))

def load_data(holdout = 0.2):
    my_data = np.genfromtxt('dataSet.csv', delimiter=',', skip_header = 1)
    tI = 1-holdout
    y = my_data[:tI*len(my_data),0]
    y_holdout = my_data[tI*len(my_data):,0] 
    X =  my_data[:tI*len(my_data),1:]
    X_holdout =  my_data[tI*len(my_data):,1:]
    return X,y,X_holdout, y_holdout


if __name__ == '__main__':

    np.random.seed(0)  # seed to shuffle the train set

    n_folds = 4
    verbose = True
    shuffle = True

    X, y, X_hold, y_hold = load_data()

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    skf = list(StratifiedKFold(y, n_folds))

    clfs = [RandomForestClassifier(n_estimators=50, n_jobs=-1, criterion='gini'),
            SVC(probability = True),
            SVC(C=100,kernel = 'rbf',probability = True),
            SVC(C=400, gamma = 0.001, kernel = 'rbf',probability = True),
            SVC(C=400, gamma = 0.00001,kernel = 'rbf',probability = True),
            SVC(C=800, gamma = 0.00001,kernel = 'rbf',probability = True),
            SVC(C=300, gamma = 0.00001,kernel = 'rbf',probability = True),
            SVC(C=1000, gamma = 0.00001,kernel = 'rbf',probability = True),
            SVC(C=10, kernel = 'rbf',probability = True),
            SVC(C=5000, kernel = 'rbf',probability = True),
            SVC(C=4000, kernel = 'rbf',probability = True),
            SVC(C=10000, kernel = 'rbf',probability = True),
            SVC(C=2000, kernel = 'rbf',probability = True),
            SVC(C=2000, kernel = 'rbf',probability = True),
            LogisticRegression(),
            RandomForestClassifier(n_estimators=10, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=25, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=30, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=35, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=40, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=20, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=10, n_jobs=-1),
            ExtraTreesClassifier(n_estimators=20, n_jobs=-1),
            ExtraTreesClassifier(n_estimators=30, n_jobs=-1),
            ExtraTreesClassifier(n_estimators=50, n_jobs=-1),
            ExtraTreesClassifier(n_estimators=70, n_jobs=-1),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1),
            MLPClassifier(),
            SVC(C=100,kernel = 'sigmoid',probability = True),
            SVC(C=400, gamma = 0.001, kernel = 'sigmoid',probability = True),
            SVC(C=400, gamma = 0.00001,kernel = 'sigmoid',probability = True),
            SVC(C=10, kernel = 'sigmoid',probability = True),
            SVC(kernel = 'sigmoid',probability = True),
            SVC(kernel = 'linear',probability = True),
            ]

    print "Creating train and test sets for blending."

    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_hold.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((X_hold.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print "Fold", i
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:, 1]
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_hold)[:, 1]
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

    print
    print "Blending."
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict(dataset_blend_test)
    
    print accuracy(y_submission,y_hold)