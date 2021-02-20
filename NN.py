import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import time

class ANN:
    def __init__(self, hyperparams=None):
        self.cross_val_score = 0
        self.learning_time = 0
        self.hyperparams = hyperparams
    def grid_search(self, X, y, nfolds):
        # create a dictionary of all values we want to test
        model = MLPClassifier( activation='relu', solver='sgd')

        print(model.get_params().keys())

        #Used this url as starting point: https://panjeh.medium.com/scikit-learn-hyperparameter-optimization-for-mlpclassifier-4d670413042b
        param_grid = {
            'hidden_layer_sizes':  [(40), (50), (60), (70), (80), (90), (100), (125), (150), (160)],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'solver': ['sgd', 'lbfgs', 'adam'],
            'max_iter':[500, 1000, 1500, 2000, 2500]
        }

        print("param grid type:", type(param_grid))


        # use gridsearch to test all values
        gscv = GridSearchCV(model, param_grid, cv=nfolds)
        #GridSearchCV()
        # fit model to data
        gscv.fit(X, y)
        print("Best params:", gscv.best_params_)
        return gscv

    def train(self, X_train, y_train, nfolds):
        kfold = StratifiedKFold(n_splits=nfolds, random_state=4, shuffle=True, )

        if self.hyperparams is not None:
            clf = MLPClassifier()
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
        self.cross_val_score = results_skfold.mean()

        print(results_skfold)
        return clf

    def test_model(self, X_train, y_train, X_test, y_test, clf):
        y_pred_test = clf.predict(X_test)
        print("ANN Accuracy")
        print(accuracy_score(y_test, y_pred_test))
        cv = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
        scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
        print("Cross Validation Scores", scores.mean())

    def do_modeling(self, X_train, y_train, X_test, y_test):

        #X = data.iloc[:, 0:-1]
        #y = data.iloc[:, -1]
        clf = self.train(X_train,y_train,nfolds=5)
        #self.test_model(X_train, y_train, X_test, y_test, clf)
        return clf

'''data = pd.read_csv('sat.trn', delimiter=' ')
X = data.iloc[:, 1:-1]
y =data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

ann = ANN()
clf = ann.do_modeling(X_train, y_train, X_test, y_test)

ypred = clf.predict(X_test)
print(classification_report(y_test,ypred))
'''