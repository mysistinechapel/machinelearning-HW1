from DecisionTree import *
from KNN import *
from SVM import *
from Boosting import *
from NN import *
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.model_selection import ShuffleSplit
from mlxtend.evaluate import bias_variance_decomp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
#borrowed from
def plot_learning_curve(estimator, X,y,n_jobs, cv, title, plot_learning_curve=True,
                        plot_n_samples_v_fit_times=True, plot_fit_times_v_score=True):
    train_sizes = np.linspace(.1, 1.0, 5)
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes, scoring='accuracy',
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    print("Plotting Learning Curve")

    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    #print("test scores mean", test_scores_mean)

    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    _, axes = plt.subplots(1, 3, figsize=(20, 5))
    # Plot learning curve
    if plot_learning_curve == True:
        axes[0].grid()
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
        axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
        axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
        axes[0].legend(loc="best")
        axes[0].set_title(title)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Accuracy")

    if plot_n_samples_v_fit_times == True:
        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-')
        axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                             fit_times_mean + fit_times_std, alpha=0.1)
        axes[1].set_xlabel("Training examples")
        #axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")

    if plot_fit_times_v_score == True:
        # Plot fit_time vs score
        axes[2].grid()
        axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
        axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1)
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")

    return plt

def plot_validation_curve(clf,X,y, param_name, param_range, title="Validation Curve with KNN Classifier",
                          xlabel="Number of Neighbours"):
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)
    #X, y = X[indices], y[indices]

    train_scores, test_scores = validation_curve(
        clf, X, y, param_name=param_name, cv=5,
        param_range=param_range, #n_jobs=psutil.cpu_count(),
        scoring="accuracy", n_jobs=4)

    train_scores_mean = np.mean(train_scores, axis=1)
    #print("Train scores:", train_scores_mean)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    #print("Test scores:", test_scores_mean)

    test_scores_std = np.std(test_scores, axis=1)
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    # plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.plot(param_range, train_scores_mean, label="Training score",
             color="blue")
    # plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="green")

    plt.legend(loc="best")
    plt.xticks(param_range)
    plt.rcParams.update({'font.size': 22})
    plt.rc('font', size=20)  # controls default text size
    plt.rc('axes', titlesize=20)  # fontsize of the title
    plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=20)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=20)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=20)  # fontsize of the legend
    return plt

def plot_roc_curve(clf,X,y):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        clf.fit(X[train], y[train])
        viz = plot_roc_curve(clf, X[test], y[test],
                             name='ROC fold {}'.format(i),
                             alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic example")
    ax.legend(loc="lower right")
#Borrowed from
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py
def plot_validation_curve_old(clf,X,y, param_name, param_range, title="Validation Curve with KNN Classifier",
                          xlabel="Number of Neighbours"):

    # Calculate accuracy on training and test set using the
    # gamma parameter with 5-fold cross validation
    plt.show()
    train_scores, test_scores = validation_curve(
        clf, X, y, param_name=param_name, param_range=param_range,
        scoring="accuracy", n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    #plt.figure(figsize=(3,3))
    plt.xlabel(xlabel)
    plt.ylabel("Score")
    #plt.ylim(0.0, 1.1)
    #plt.xticks(param_range, param_range)
    lw = 10
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="blue", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="blue", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="red", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="red", lw=lw)
    plt.legend(loc="best")
    plt.xticks(param_range)


    plt.show()
    return plt

def decision_tree(X_train_normed, y_train, X_test_normed, y_test, dt_param_grid, dataset_name):
    dt = DecisionTree(data, fields_to_categorize=None, hyperparams=dt_param_grid)
    model = dt.do_modeling(X_train_normed, y_train, X_test_normed, y_test)
    #print("Bias Variance for tuned  " + dataset_name)
    #bias_variance(model, X_train_normed, y_train, X_test_normed, y_test)
    initial_dt = DecisionTreeClassifier()
    #print("Bias Variance for initial  " + dataset_name)
    #bias_variance(initial_dt, X_train_normed, y_train, X_test_normed, y_test)

    initial_dt.fit(X_train_normed, y_train)
    plt = plot_learning_curve(initial_dt, X_train_normed, y_train, 5, cv, "Decision Tree Learning Curve")
    plt.savefig("images/" + dataset_name + "_Initial_DT_Learning_Curve.png", bbox_inches='tight')
    plt.close()

    #plot_roc_curve(model, X_train, y_train)
    plt = plot_learning_curve(model, X_train_normed, y_train, 5, cv, "Decision Tree Learning Curve")
    plt.savefig("images/" + dataset_name + "_DT_Learning_Curve.png")
    plt.close()

    # Leave these commented out
    '''parameter_range = ['gini', 'entropy']
    plt_criterion = plot_validation_curve(DecisionTreeClassifier(), X_train, y_train, "criterion", parameter_range,
                                      title="Validation Curve with Decision Tree Classifier - Criterion", xlabel="Criterion")
    plt_criterion.savefig("DT_Val_Curve_criterion")
    plt_criterion.close()


    parameter_range = ['best', 'random']
    val_curve = plot_validation_curve(DecisionTreeClassifier(), X_train, y_train, "splitter", parameter_range,
                                      title="Validation Curve with Decision Tree Classifier - Splitter", xlabel="Splitter")
    val_curve.savefig("DT_Val_Curve_splitter")
    val_curve.close()'''
    #
    parameter_range = np.arange(2, 15, 5)
    val_curve = plot_validation_curve(DecisionTreeClassifier(), X_train_normed, y_train, "max_depth", parameter_range,
                                      title="Validation Curve with Decision Tree - Max Depth", xlabel="Max Depth")
    val_curve.savefig("images/" + dataset_name + "_DT_Val_Curve_max_depth.png")
    val_curve.close()

    parameter_range = [.0001, .001, .01]
    val_curve = plot_validation_curve(DecisionTreeClassifier(), X_train_normed, y_train, "ccp_alpha",
                                      parameter_range,
                                      title="Decision Tree Validation Curve", xlabel="ccp alpha")
    val_curve.savefig("images/" + dataset_name + "_DT_Val_Curve_ccp_alpha.png")
    val_curve.close()

    parameter_range = np.arange(5, 75, 5)
    val_curve = plot_validation_curve(DecisionTreeClassifier(), X_train_normed, y_train, "max_leaf_nodes",
                                      parameter_range,
                                      title="Validation Curve with Decision Tree", xlabel="Max Leaf Nodes")
    val_curve.savefig("images/" + dataset_name + "_DT_Val_Curve_max_leaf_nodes.png")
    plt.close()
    # test model
    begin = time.time()
    y_test_pred = model.predict(X_test_normed)
    end = time.time()

    query_time = end - begin
    test_score = accuracy_score(y_test, y_test_pred)
    train_cross_val = dt.cross_val_score
    learning_time = dt.learning_time
    return query_time, test_score, train_cross_val, learning_time, model





def ann(X_train_normed, y_train, X_test_normed, y_test, param_grid, dataset_name):
    print("Running experiments for Neural Networks")
    ann = ANN(hyperparams=param_grid)
    model = ann.do_modeling(X_train_normed, y_train, X_test_normed, y_test)

    print("Plotting ann learning curve")
    plt = plot_learning_curve(model, X_train_normed, y_train, 5, cv, "ANN Learning Curve")
    plt.savefig("images/" + dataset_name + "_ann_learning_curve.png")

    print("plotting val curve")
    parameter_range = [(25), (200), (500), (1000), (1500), (2000)]
    val_curve = plot_validation_curve(MLPClassifier(), X_train_normed, y_train, "hidden_layer_sizes", parameter_range,
                                      title="ANN Validation Curve", xlabel="Hidden Layer Sizes")
    val_curve.savefig("images/" + dataset_name + "ANN_Val_Curve_hidden_layer_sizes_one.png")
    val_curve.close()


    print("Plotting ann val curve for max iter")
    parameter_range = [50, 1000,5000,10000, 50000, 100000]
    val_curve = plot_validation_curve(MLPClassifier(), X_train_normed, y_train, "max_iter", parameter_range,
                                      title="ANN Validation Curve", xlabel="Max Iterations")
    val_curve.savefig("images/" + dataset_name + "ANN_Val_Curve_max_iter.png")
    val_curve.close()

    print("Plotting ann val curve for solver")
    parameter_range = ['sgd', 'lbfgs', 'adam']
    val_curve = plot_validation_curve(MLPClassifier(), X_train_normed, y_train, "solver", parameter_range,
                                      title="ANN Validation Curve", xlabel="Solver")
    val_curve.savefig("images/" + dataset_name + "ANN_Val_Curve_solver.png")
    val_curve.close()

    print("Plotting ann val curve for learning rate")
    parameter_range = ['constant', 'invscaling', 'adaptive']
    val_curve = plot_validation_curve(MLPClassifier(), X_train_normed, y_train, "learning_rate", parameter_range,
                                      title="ANN Validation Curve", xlabel="Learning Rate")
    val_curve.savefig("images/" + dataset_name + "ANN_Val_Curve_learning_rate.png")
    val_curve.close()

    print("ANN - Estimating wallclock time for predicting")
    begin = time.time()
    y_test_pred = model.predict(X_test_normed)
    end = time.time()

    query_time = end - begin
    test_score = accuracy_score(y_test, y_test_pred)
    train_cross_val = ann.cross_val_score
    learning_time = ann.learning_time
    return query_time, test_score, train_cross_val, learning_time

def boosting(X_train_normed, y_train, X_test_normed, y_test, param_grid, dataset_name, dt_model):
    print("Boosting Experiments")
    boost = Boosting(hyperparams=param_grid, base_estimator=dt_model)
    model = boost.do_modeling(X_train_normed, y_train, X_test_normed, y_test)
    plt = plot_learning_curve(model, X_train_normed, y_train, 5, cv, "Boosting Learning Curve")
    plt.savefig("images/" + dataset_name + "_boosting_learning_curve.png")
    plt.close()

    parameter_range = [.002, .5, .75, 1]
    val_curve = plot_validation_curve(GradientBoostingClassifier(n_estimators=350), X_train_normed, y_train, "learning_rate",
                                      parameter_range,
                                      title="Boosting Validation Curve", xlabel="learning_rate")
    val_curve.savefig("images/" + dataset_name + "_Boosting_Val_Curve_learning_rate.png")
    val_curve.close()
    parameter_range = [1, 500,1000,1500,2000]
    val_curve = plot_validation_curve(GradientBoostingClassifier(), X_train_normed, y_train, "n_estimators", parameter_range,
                                      title="Boosting Validation Curve", xlabel="Estimators")
    val_curve.savefig("images/" + dataset_name + "Boosting_Val_Curve_Estimators.png")
    val_curve.close()

    parameter_range = np.arange(2, 10)
    val_curve = plot_validation_curve(GradientBoostingClassifier(), X_train_normed, y_train, "max_features", parameter_range,
                                      title="Boosting Validation Curve", xlabel="Max Features")
    val_curve.savefig("images/" + dataset_name + "Boosting_Val_Curve_Max_Features.png")
    val_curve.close()

    parameter_range = np.arange(2, 15)
    val_curve = plot_validation_curve(GradientBoostingClassifier(), X_train_normed, y_train, "max_depth", parameter_range,
                                      title="Boosting Validation Curve", xlabel="Max Depth")
    val_curve.savefig("images/" + dataset_name + "Boosting_Val_Curve_Max_Depth.png")
    val_curve.close()



    begin = time.time()
    y_test_pred = model.predict(X_test_normed)
    end = time.time()

    query_time = end - begin
    test_score = accuracy_score(y_test, y_test_pred)
    train_cross_val = boost.cross_val_score
    learning_time = boost.learning_time
    return query_time, test_score, train_cross_val, learning_time

def knn(X_train_normed, y_train, X_test_normed, y_test, param_grid, dataset_name):
    print("Running KNN Experiments")
    knn = KNN(hyperparams=param_grid)
    model = knn.do_modeling(X_train_normed, y_train, X_test_normed, y_test)
    cv = knn.cv
    plt = plot_learning_curve(model, X_train_normed, y_train, 5, cv, "KNN Learning Curve")
    plt.savefig("images/" + dataset_name + "_knn_learning_curve.png")
    plt.close()

    parameter_range = np.arange(1, 200, 50)
    val_curve = plot_validation_curve(KNeighborsClassifier(n_jobs=4), X_train_normed, y_train, "n_neighbors", parameter_range,
                                      title="Validation Curve with KNN Classifier", xlabel="Number of Neighbours")
    val_curve.savefig("images/" + dataset_name + "KNN_Val_Curve_n_neighbors")
    val_curve.close()

    parameter_range = ['distance', 'uniform']
    val_curve = plot_validation_curve(KNeighborsClassifier(n_jobs=4), X_train_normed, y_train, "weights", parameter_range,
                                      title="Validation Curve with KNN Classifier", xlabel="Weights")
    val_curve.savefig("images/" + dataset_name + "KNN_Val_Curve_weights")
    val_curve.close()
    '''
    parameter_range = np.arange(1,50, 2)
    val_curve = plot_validation_curve(KNeighborsClassifier(n_jobs=4), X_train_normed, y_train, "leaf_size", parameter_range,
                                      title="Validation Curve with KNN Classifier", xlabel="Leaf Size")
    val_curve.savefig("images/" + dataset_name + "KNN_Val_Curve_leaf_size")
    val_curve.close()

    '''

    parameter_range = ['manhattan', 'euclidean']
    val_curve = plot_validation_curve(KNeighborsClassifier(n_jobs=4), X_train_normed, y_train, "metric", parameter_range,
                                      title="Validation Curve with KNN Classifier", xlabel="Distance Metric")
    val_curve.savefig("images/" + dataset_name + "KNN_Val_Curve_metric")
    val_curve.close()

    begin = time.time()
    y_test_pred = model.predict(X_test_normed)
    end = time.time()

    query_time = end - begin
    test_score = accuracy_score(y_test, y_test_pred)
    train_cross_val = knn.cross_val_score
    learning_time = knn.learning_time
    return query_time, test_score, train_cross_val, learning_time

def svm(X_train_normed, y_train, X_test_normed, y_test, param_grid, dataset_name, kernel):
    svc = SVM(kernel=kernel, hyperparams=param_grid)
    model = svc.do_modeling(X_train_normed, y_train, X_test_normed, y_test)
    plt = plot_learning_curve(model, X_train_normed, y_train, 5, cv, "SVM Learning Curve")
    plt.savefig("images/" + dataset_name + kernel + "_SVM_Learning_Curve.png")
    plt.close()

    if kernel == 'poly':
        parameter_range = np.arange(0, 6)
        val_curve = plot_validation_curve(SVC(kernel=kernel), X_train_normed, y_train, "degree", parameter_range,
                                          title="SVM Validation Curve", xlabel="Degree")
        val_curve.savefig("images/" + dataset_name + kernel + "_SVM_Degree")
        val_curve.close()

    parameter_range = np.arange(.01, 30, 5)
    val_curve = plot_validation_curve(SVC(kernel=kernel, class_weight='balanced'), X_train_normed, y_train, "C", parameter_range,
                                      title="SVM Validation Curve", xlabel="C")
    val_curve.savefig("images/" + dataset_name + kernel + "_SVM_Val_Curve_C_unweighted")
    val_curve.close()

    parameter_range = np.arange(.01, 30, 5)
    val_curve = plot_validation_curve(SVC(kernel=kernel), X_train_normed, y_train, "C", parameter_range,
                                      title="SVM Validation Curve", xlabel="C")
    val_curve.savefig("images/" + dataset_name + kernel + "_SVM_Val_Curve_C_weighted")
    val_curve.close()

    '''parameter_range = ['balanced', None]
    val_curve = plot_validation_curve(SVC(kernel=kernel), X_train_normed, y_train, "class_weight",
                                      parameter_range,
                                      title="SVM Validation Curve", xlabel="class_weight")
    val_curve.savefig("images/" + dataset_name + kernel + "_SVM_Val_Curve_class_weight")
    val_curve.close()'''

    if kernel == "poly":
        parameter_range = ["scale", "auto"]
        val_curve = plot_validation_curve(SVC(kernel=kernel), X_train_normed, y_train, "gamma",
                                          parameter_range,
                                          title="SVM Validation Curve", xlabel="gamma")
        val_curve.savefig("images/" + dataset_name + kernel + "_SVM_Val_Curve_gamma")
        val_curve.close()

    begin = time.time()
    y_test_pred = model.predict(X_test_normed)
    end = time.time()

    query_time = end - begin
    test_score = accuracy_score(y_test, y_test_pred)
    train_cross_val = svc.cross_val_score
    learning_time = svc.learning_time

    '''svc = SVM(kernel="rbf", hyperparams=rbf_param_grid)
    model = svc.do_modeling(X_train_normed, y_train, X_test_normed, y_test)
    plt = plot_learning_curve(model, X_train_normed, y_train, 5, cv, "SVM Learning Curve")
    plt.savefig("images/" + dataset_name + "rbf_svm_learning_curve.png")
    plt.close()

    parameter_range = np.arange(1,16,3)
    val_curve = plot_validation_curve(SVC(kernel="rbf"), X_train_normed, y_train, "C", parameter_range,
                                      title="SVM Validation Curve", xlabel="C")
    val_curve.savefig("images/" + dataset_name + "RBF_SVM_Val_Curve_C")
    val_curve.close()

    begin = time.time()
    y_test_pred = model.predict(X_test_normed)
    end = time.time()

    query_time_rbf = end - begin
    test_score_rbf = accuracy_score(y_test, y_test_pred)
    train_cross_val_rbf = svc.cross_val_score
    learning_time_rbf = svc.learning_time'''
    return query_time, test_score, train_cross_val, learning_time#, \
      #query_time_rbf, test_score_rbf, train_cross_val_rbf, learning_time_rbf

#borrowed from https://machinelearningmastery.com/calculate-the-bias-variance-trade-off/
def bias_variance(model, X_train, y_train, X_test, y_test ):
    print(type(X_train))
    '''if type(X_train) != "numpy.ndarray":
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()
        X_test = X_test.to_numpy()
        y_test = y_test.to_numpy()'''

    mse, bias, var = bias_variance_decomp(model, X_train, y_train, X_test, y_test, loss='mse', num_rounds=200,
                                          random_seed=1)


    print('MSE: %.3f' % mse)
    print('Bias: %.3f' % bias)
    print('Variance: %.3f' % var)
filename = "../titanic/arrhythmia.data"
filename = "sat.trn"
#filename = "pima_diabetes.csv"
'''data = pd.read_csv("mushroom.data")
print(data.shape)
# categorize adult data
indexes_of_cat = np.arange(0, 23)
for i in range(len(indexes_of_cat)):
    print(indexes_of_cat[i])
    data.iloc[:, indexes_of_cat[i] - 1] = data.iloc[:, indexes_of_cat[i] - 1].astype('category')
    data.iloc[:, indexes_of_cat[i] - 1] = data.iloc[:, indexes_of_cat[i] - 1].cat.codes
'''
#print(data)
#X = data.iloc[:, 0:-1]
#y = data.iloc[:, -1]
#data = pd.read_csv(filename, delimiter=' ')
data = pd.read_csv("adult_new.csv")
# categorize adult data
indexes_of_cat = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15]
for i in range(len(indexes_of_cat)):
    print(indexes_of_cat[i])
    data.iloc[:, indexes_of_cat[i] - 1] = data.iloc[:, indexes_of_cat[i] - 1].astype('category')
    data.iloc[:, indexes_of_cat[i] - 1] = data.iloc[:, indexes_of_cat[i] - 1].cat.codes

print(data)
X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]
#X = data.iloc[:, 0:-1]
#y = data.iloc[:, -1]
X_train_adult, X_test_adult, y_train_adult, y_test_adult = train_test_split(X, y, test_size=0.3,stratify=y)
scaler = preprocessing.StandardScaler().fit(X_train_adult)
X_train_normed_adult = scaler.transform(X_train_adult) #scaled
X_test_normed_adult = scaler.transform(X_test_adult)
y_train_adult = y_train_adult.to_numpy()
y_test_adult = y_test_adult.to_numpy()
cv = RepeatedStratifiedKFold(n_splits=5, random_state=900, n_repeats=3 )
dt_param_grid = {'criterion': 'gini',
                 #'ccp_alpha':.001
                      'max_depth':6,
                      #'splitter':['best', 'random'],
                      #'min_samples_split': [2],
                      #'max_leaf_nodes': [25, 30],
                      #'max_leaf_nodes': 63
                      }
#
query_time1, test_score1, train_cross_val1, learning_time1, adult_dt = decision_tree(X_train_normed_adult, y_train_adult, X_test_normed_adult, y_test_adult, dt_param_grid, "Adult")


data = pd.read_csv("synthetic_data.csv")
print(data)
X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True)
cv = StratifiedKFold(n_splits=5, random_state=1127, shuffle=True )

X_train_synthetic, X_test_synthetic, y_train_synthetic, y_test_synthetic = train_test_split(X, y, test_size=0.3,stratify=y)
scaler = preprocessing.StandardScaler().fit(X_train_synthetic)
X_train_normed_synthetic = scaler.transform(X_train_synthetic) #scaled
X_test_normed_synthetic = scaler.transform(X_test_synthetic)
cv = StratifiedKFold(n_splits=5, random_state=1127, shuffle=True )

dt_param_grid = {'criterion': 'entropy',
                      #'ccp_alpha':.0005,
                      'max_depth':7
                      #'splitter':['best', 'random'],
                      #'min_samples_split': [2],
                      #'max_leaf_nodes': [25, 30],
                      #'max_leaf_nodes': 70
                      }
#replace param grid with None

print("Printing Decision Tree Stats")

query_time2, test_score2, train_cross_val2, learning_time2, synth_dt = decision_tree(X_train_normed_synthetic, y_train_synthetic, X_test_normed_synthetic, y_test_synthetic, dt_param_grid, "Synthetic")
list = [['Adult', learning_time1, query_time1, train_cross_val1,test_score1],
        ['Synthetic', learning_time2, query_time2, train_cross_val2,test_score2]]
columns=['DATASET','TRAIN TIME', 'PREDICT TIME', 'CV SCORE', 'TEST SCORE']
df = pd.DataFrame(list, columns=columns)
print(df)
print("Printing ANN stats")
adult_ann_grid  = {
            'hidden_layer_sizes':  (40),
            'learning_rate': 'adaptive',
            'solver': 'sgd',
            'max_iter':25
        }

synthetic_ann_grid  = {
            'hidden_layer_sizes':  (125),
            'learning_rate': 'adaptive',
            'solver': 'sgd', #sgd
            'max_iter':300
        }
query_time1, test_score1, train_cross_val1, learning_time1 = ann(X_train_normed_adult, y_train_adult, X_test_normed_adult, y_test_adult, adult_ann_grid, "Adult")
query_time2, test_score2, train_cross_val2, learning_time2 = ann(X_train_normed_synthetic, y_train_synthetic, X_test_normed_synthetic, y_test_synthetic, synthetic_ann_grid, "Synthetic")

list = [['Adult', learning_time1, query_time1, train_cross_val1,test_score1],
        ['Synthetic', learning_time2, query_time2, train_cross_val2,test_score2]]
columns=['DATASET','TRAIN TIME', 'PREDICT TIME', 'CV SCORE', 'TEST SCORE']
df = pd.DataFrame(list, columns=columns)
print(df)
print("Printing linerar svm stats")
adult_poly_svm_grid  = {
            'C':  5.0,
            'class_weight': 'balanced',
            'degree':3
        }

adult_rbf_svm_grid  = {
            'C':  3.51,
            'gamma': 'auto',
            'class_weight': 'balanced'
        }

#first variabes are for adult dataset
query_time1_rbf, test_score1_rbf, train_cross_val1_rbf, learning_time1_rbf = svm(X_train_normed_adult, y_train_adult, X_test_normed_adult, y_test_adult, adult_rbf_svm_grid, "Adult", "rbf")
query_time1_poly, test_score1_poly, train_cross_val1_poly, learning_time1_poly = svm(X_train_normed_adult, y_train_adult, X_test_normed_adult, y_test_adult, adult_poly_svm_grid, "Adult", "poly")

synthetic_poly_svm_grid  = {
            'C':  5,#,
            'degree':3
            #'class_weight': 'balanced'
        }

synthetic_rbf_grid  = {
            'C':  6.51,
            'gamma': 'auto',
            'class_weight': 'balanced'
        }
query_time2_rbf, test_score2_rbf, train_cross_val2_rbf, learning_time2_rbf = svm(X_train_normed_synthetic, y_train_synthetic, X_test_normed_synthetic, y_test_synthetic,  synthetic_rbf_grid, "Synthetic", "rbf")
query_time2_poly, test_score2_poly, train_cross_val2_poly, learning_time2_poly = svm(X_train_normed_synthetic, y_train_synthetic, X_test_normed_synthetic, y_test_synthetic,  synthetic_poly_svm_grid, "Synthetic", "poly")

list = [['Adult - Poly SVM', learning_time1_poly, query_time1_poly, train_cross_val1_poly,test_score1_poly],
        ['Adult - RBF SVM', learning_time1_rbf, query_time1_rbf, train_cross_val1_rbf,test_score1_rbf],
        ['Synthetic - Poly  SVM', learning_time2_poly, query_time2_poly, train_cross_val2_poly,test_score2_poly],
        ['Synthetic - RBF SVM', learning_time2_rbf, query_time2_rbf, train_cross_val2_rbf,test_score2_rbf]]

columns=['DATASET','TRAIN TIME', 'PREDICT TIME', 'CV SCORE', 'TEST SCORE']
df = pd.DataFrame(list, columns=columns)
print(df)

print("Printing boosting Stats")
snythetic_boost_params = {'n_estimators':500,
                          #'ccp_alpha':.001,
                          'max_depth':4,
                          'learning_rate':0.05
                          }
                          #'estimators':250#,

                      #'splitter':['best', 'random'],
                      #'min_samples_split': [2],
                      #'max_leaf_nodes': [25, 30],
                      #'max_leaf_nodes': 70

adult_boost_params = {'n_estimators':500,
                      'max_depth':3,
                      #'ccp_alpha':.001,
                      'learning_rate':0.10
                      }
                      #'learning_rate':0.15,
                      #'estimators':110
                      #'min_child_weight': 1,
                      #'gamma':0,
                      #'scale_pos_weight':1,
                      #'min_samples_split': [2],
                      #'max_leaf_nodes': [25, 30],
                      #'max_leaf_nodes': 70

query_time2, test_score2, train_cross_val2, learning_time2 = boosting(X_train_normed_adult, y_train_adult, X_test_normed_adult, y_test_adult, adult_boost_params, "Adult", adult_dt)
query_time1, test_score1, train_cross_val1, learning_time1 = boosting(X_train_normed_synthetic, y_train_synthetic, X_test_normed_synthetic, y_test_synthetic, snythetic_boost_params, "Synthetic", synth_dt)


list = [['Adult', learning_time1, query_time1, train_cross_val1,test_score1],
        ['Synthetic', learning_time2, query_time2, train_cross_val2,test_score2]]
columns=['DATASET','TRAIN TIME', 'PREDICT TIME', 'CV SCORE', 'TEST SCORE']
df = pd.DataFrame(list, columns=columns)
print(df)


print("Printing KNN Stats")
knn_param_grid_adult = {'n_neighbors': 31, 'weights':'uniform'
                      }
knn_param_grid_synthetic = {'n_neighbors': 23, 'weights':'uniform'
                      }
query_time1, test_score1, train_cross_val1, learning_time1 = knn(X_train_normed_adult, y_train_adult, X_test_normed_adult, y_test_adult, knn_param_grid_adult, "Adult")
query_time2, test_score2, train_cross_val2, learning_time2 = knn(X_train_normed_synthetic, y_train_synthetic, X_test_normed_synthetic, y_test_synthetic, knn_param_grid_synthetic, "Synthetic")
list = [['Adult', learning_time1, query_time1, train_cross_val1,test_score1],
        ['Synthetic', learning_time2, query_time2, train_cross_val2,test_score2]]
columns=['DATASET','TRAIN TIME', 'PREDICT TIME', 'CV SCORE', 'TEST SCORE']
df = pd.DataFrame(list, columns=columns)
print(df)

