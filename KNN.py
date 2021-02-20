from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import GridSearchCV
import time
class KNN():

    def __init__(self, hyperparams=None):
        self.cv = None
        self.learning_time = 0
        self.cross_val_score = 0
        self.hyperparams = hyperparams
    def grid_search(self, X, y, nfolds):
        # create a dictionary of all values we want to test
        model = KNeighborsClassifier(n_jobs=4)

        print(model.get_params().keys())
        param_grid = {'n_neighbors': np.arange(2, 40),
                      'weights':['uniform']}

        # use gridsearch to test all values
        gscv = GridSearchCV(model, param_grid, cv=nfolds)
        # fit model to data
        gscv.fit(X, y)
        print("Best params:", gscv.best_params_)
        return gscv

    def train(self, X_train, y_train, nfolds):
        kfold = StratifiedKFold(n_splits=nfolds, random_state=4, shuffle=True)

        if self.hyperparams is not None:
            clf = KNeighborsClassifier(n_jobs=4)
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
        self.cv = kfold
        print(grd_search)

        clf = grd_search.best_estimator_
        begin = time.time()
        clf.fit(X_train, y_train)
        end = time.time()
        self.learning_time = end - begin
        results_skfold = cross_val_score(clf, X_train, y_train, cv=kfold)
        print("cross validation score:", results_skfold.mean())
        self.cross_val_score = results_skfold.mean()




        return clf

    def test_model(self, X_train, y_train, X_test, y_test, clf):
        print("testing KNN")
        y_pred_test = clf.predict(X_test)
        cv = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
        scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
        print("Cross Validation Scores", scores.mean())
        print("KNN TEST ACCURACY", accuracy_score(y_test, y_pred_test))


    def do_modeling(self, X_train, y_train, X_test, y_test):

        #X = data.iloc[:, 0:-1]
        #y = data.iloc[:, -1]
        clf = self.train(X_train,y_train,nfolds=5)
        self.clf = clf
        #self.test_model(X_train, y_train, X_test, y_test, clf)
        return clf
'''knn = KNN()
data = pd.read_csv("mushroom.data")
indexes_of_cat = np.arange(0, 23)
for i in range(len(indexes_of_cat)):
    print(indexes_of_cat[i])
    data.iloc[:, indexes_of_cat[i] - 1] = data.iloc[:, indexes_of_cat[i] - 1].astype('category')
    data.iloc[:, indexes_of_cat[i] - 1] = data.iloc[:, indexes_of_cat[i] - 1].cat.codes
X = data.iloc[:, 1:-1]
y = data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True, stratify=y,
                                                    random_state=5)
loo = LeaveOneOut()
clf = knn.grid_search(X_train, y_train, 5)
#clf = KNeighborsClassifier(n_neighbors=15)
clf.fit(X_train, y_train)
kfold = StratifiedShuffleSplit(n_splits=5, random_state=3)

results_skfold = cross_val_score(clf, X_train, y_train, cv=kfold)
print(results_skfold.mean())
y_predict = clf.predict(X_test)
print(classification_report(y_test,y_predict))'''
