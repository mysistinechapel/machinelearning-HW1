#ROC and AUC only supports binary classification
#Best params when using CVGridSearch could be overfitted params
#
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.svm import SVC
import time
class DecisionTree:

    #def make_charts(self):
    def __init__(self, data, fields_to_categorize=None, hyperparams=None):
        '''
        :param data: The data we are training
        :param fields_to_categorize: a list of fields by index number we need to categorize
        '''
        self.model = None
        self.cv = None
        if fields_to_categorize is not None:
            print("Categorizing Data")
            for i in range(len(fields_to_categorize)):
                data.iloc[:,i] = data.iloc[:,i].astype('category')
                data.iloc[:,i] = data.iloc[:,i].cat.codes

        self.data = data
        self.hyperparams = hyperparams
        self.learning_time = 0
        self.cross_val_score = 0

    def train_dt(self):

        best_params = self.dtree_grid_search(X, y, 5)

    def dtree_grid_search(self,X, y, nfolds):
        # create a dictionary of all values we want to test
        param_grid = {'criterion': ['entropy', 'gini'],
                      'max_depth': np.arange(1,14, 1),
                      #'splitter':['best', 'random'],
                      #'min_samples_split': np.arange(2,30, 5),
                      'ccp_alpha': [.001, .002, .0030, .004, .005],
                      'max_leaf_nodes': np.arange(2,70,10)
                      }

        # decision tree model
        dtree_model = DecisionTreeClassifier(random_state=1127)
        # use gridsearch to test all values
        dtree_gscv = GridSearchCV(dtree_model, param_grid, cv=nfolds)
        # fit model to data
        dtree_gscv.fit(X, y)
        print("Best params:", dtree_gscv.best_params_)
        return dtree_gscv

    def train(self, X_train, y_train, nfolds=7):
        print("Training Decision Tree")
        kfold = RepeatedStratifiedKFold(n_splits=nfolds, random_state=900,  n_repeats=3)
        if self.hyperparams is not None:
            clf = DecisionTreeClassifier()
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
        grd_search = self.dtree_grid_search(X_train,y_train,nfolds=5)
        print(grd_search)

        clf = grd_search.best_estimator_
        clf.fit(X_train, y_train)
        results_skfold = cross_val_score(clf, X_train, y_train, cv=kfold)
        self.cv = kfold #I will use this when doing experiments to ensure cv object is consistent
        print("Cross Validation Score:", results_skfold)

        #X = X.loc[:, ~(X == '?').any()]
        #X_train.to_csv("test.csv")
        #clf.fit(X_train, y_train)


        return clf

    def test_model(self, X_train, y_train, X_test, y_test, clf):
        print("testing model - x_test shape:", X_test.shape)
        y_pred_test = clf.predict(X_test)
        print("DT Accuracy")
        print(accuracy_score(y_test, y_pred_test))
        cv = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
        scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
        print("Cross Validation Scores", scores.mean())

        #print("Accuracy: %.2f%%" % (results_skfold.mean() * 100.0))
    def do_modeling(self, X_train, y_train, X_test, y_test):

        #X = data.iloc[:, 0:-1]
        #y = data.iloc[:, -1]
        clf = self.train(X_train,y_train,nfolds=5)
        self.test_model(X_train, y_train, X_test, y_test, clf)
        return clf

'''
filename = "../titanic/arrhythmia.data"
filename = "sat.trn"
data = pd.read_csv("adult.data")

#filename = 'hcv.csv'
indexes_of_cat = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15]
for i in range(len(indexes_of_cat)):
    print(indexes_of_cat[i])
    data.iloc[:, indexes_of_cat[i] - 1] = data.iloc[:, indexes_of_cat[i] - 1].astype('category')
    data.iloc[:, indexes_of_cat[i] - 1] = data.iloc[:, indexes_of_cat[i] - 1].cat.codes

print(data)
X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y)
'''


#dt = DecisionTree(data)
#clf = dt.do_modeling(X_train, y_train, X_test, y_test)

#ypred = clf.predict(X_test)
#print(classification_report(y_test,ypred))



#to do
#1) ROC Curve
#2) AUC Curve
#3) Cross Validation
#4) https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py