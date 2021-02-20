import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import time

class SVM:
    def __init__(self, kernel="linear",hyperparams=None):
        self.kernel=kernel
        self.learning_time = 0
        self.cross_val_score = 0
        self.hyperparams = hyperparams

    def grid_search(self, X, y, nfolds):
        # create a dictionary of all values we want to test
        model = SVC(kernel=self.kernel)

        print(model.get_params().keys())
        param_grid = {
                      'C': np.arange(.01, 7, .5),
                      'class_weight':['balanced', None],
                      'gamma': ['scale', 'auto']
                      }

        # use gridsearch to test all values
        gscv = GridSearchCV(model, param_grid, cv=nfolds)
        # fit model to data
        gscv.fit(X, y)

        print("Best params:", gscv.best_params_)
        return gscv

    def train(self, X_train, y_train, nfolds):
        kfold = StratifiedKFold(n_splits=nfolds, random_state=4, shuffle=True, )

        if self.hyperparams is not None:
            clf = SVC(kernel=self.kernel)
            print(self.hyperparams)
            clf.set_params(**self.hyperparams)
            begin = time.time()
            clf.fit(X_train, y_train)
            end = time.time()
            self.learning_time = end - begin
            results_skfold = cross_val_score(clf, X_train, y_train, cv=kfold)
            print("Cross Validation Score:", results_skfold)
            self.cross_val_score = results_skfold.mean()

            return clf
        grd_search = self.grid_search(X_train, y_train, nfolds=5)
        print(grd_search)

        clf = grd_search.best_estimator_
        begin = time.time()
        clf.fit(X_train, y_train)
        end = time.time()

        self.learning_time = end - begin
        results_skfold = cross_val_score(clf, X_train, y_train, cv=kfold)
        print(results_skfold)
        self.cross_val_score = results_skfold.mean()
        return clf

    def test_model(self, X_train, y_train, X_test, y_test, clf):
        y_pred_test = clf.predict(X_test)
        print("SVM Accuracy")
        print(accuracy_score(y_test, y_pred_test))
        cv = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
        scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
        print("Cross Validation Scores", scores.mean())

    def do_modeling(self, X_train, y_train, X_test, y_test):

        #X = data.iloc[:, 0:-1]
        #y = data.iloc[:, -1]
        clf = self.train(X_train,y_train,nfolds=5)
        self.test_model(X_train, y_train, X_test, y_test, clf)
        return clf

'''data = pd.read_csv('sat.trn', delimiter=' ')
X = data.iloc[:, 1:-1]
y =data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

clf = SVC()
clf.fit(X_train, y_train)
kfold = StratifiedKFold(n_splits=9, random_state=3, shuffle=True )
results_skfold = cross_val_score(clf, X_train, y_train, cv=kfold)
print(results_skfold.mean())

ypred = clf.predict(X_test)
print(classification_report(y_test,ypred))'''
