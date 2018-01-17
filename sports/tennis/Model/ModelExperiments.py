import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from xgboost import XGBClassifier


def logloss(y_true, y_probas):
    """This function must return a numerical value given two numpy arrays 
    containing the ground truth labels and generated meta-features, in that order.
    (In this example, `y_true` and `y_probas`)
    """
    from sklearn.metrics import log_loss
    return log_loss(y_true, y_probas)



if __name__ == '__main__':

    np.random.seed(0)  # seed to shuffle the train set

    df = pd.read_csv('/home/jbl/Documents/Programming/Sports Betting/tennis/Data/FeatureSet.csv', index_col = 'ix')



    target = df['Target']
    df = df.drop(['Target'], axis = 1)

    X = df[:-17]
    y = target[:-17]

    X_submission = df[-17:]
   # X_submission = X_submission.drop(['Target'], axis = 1)

    n_folds = 5
    verbose = True
    shuffle = False

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    skf = list(StratifiedKFold(y, n_folds))

    clfs = [RandomForestClassifier(random_state=8, max_depth = 2, n_estimators = 100),
            RandomForestClassifier(random_state=8, max_depth = 10, n_estimators = 100),
            XGBClassifier(seed=8, max_depth = 10),
            LogisticRegression(),
            XGBClassifier(seed=8, max_depth = 2)
            ]

    print "Creating train and test sets for blending."

    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print("Fold", i)
            X_train = X.iloc[train]
            y_train = y[train]
            X_test = X.iloc[test]
            y_test = y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:, 1]
            print("Fold Score:" , logloss(y_test,y_submission))
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:, 1]
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

    print ("Blending.")
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict_proba(dataset_blend_test)[:, 1]



    print "Saving Results."
