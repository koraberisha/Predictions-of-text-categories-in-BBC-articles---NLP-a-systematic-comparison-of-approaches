import numpy as np
import pandas as pd
import nltk
import string
import sklearn
from nltk.corpus import stopwords
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn import metrics,preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from collections import defaultdict
from pathlib import Path
import matplotlib
from matplotlib import pyplot as plt
from random import randint

matplotlib.use('qtagg')

def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

def get_stopwords():
    nltk_stopwords = set(stopwords.words('english'))

    return nltk_stopwords

def gen_training_set_tfid(x):
    tfidfvectorizer = TfidfVectorizer(max_features=1000,stop_words=get_stopwords())
    tfidfvectorizer.fit(x)
    xtrain_tfidf = tfidfvectorizer.transform(x)

    return xtrain_tfidf

def gen_training_set_count(x):
    countVec = CountVectorizer(stop_words=get_stopwords())
    countVec.fit(x)
    xtrain_count = countVec.transform(x)

    return xtrain_count

def gen_training_set_count_bigram(x):
    countVec = CountVectorizer(ngram_range =(1, 2),max_features=1000,stop_words=get_stopwords())
    countVec.fit(x)
    xtrain_count = countVec.transform(x)

    return xtrain_count

def random_forest_parameters():
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    return random_grid

def GridSearch_table_plot(grid_clf, param_name,
                          num_results=15,
                          negative=True,
                          graph=True,
                          display_all_params=True):

    from IPython.display import display
    import pandas as pd
    matplotlib.use('TkAgg')

    clf = grid_clf.best_estimator_
    clf_params = grid_clf.best_params_
    if negative:
        clf_score = -grid_clf.best_score_
    else:
        clf_score = grid_clf.best_score_
    clf_stdev = grid_clf.cv_results_['std_test_score'][grid_clf.best_index_]
    cv_results = grid_clf.cv_results_

    print("best parameters: {}".format(clf_params))
    print("best score:      {:0.5f} (+/-{:0.5f})".format(clf_score, clf_stdev))
    if display_all_params:
        import pprint
        pprint.pprint(clf.get_params())

    scores_df = pd.DataFrame(cv_results).sort_values(by='rank_test_score')

    best_row = scores_df.iloc[0, :]
    if negative:
        best_mean = -best_row['mean_test_score']
    else:
        best_mean = best_row['mean_test_score']
    best_stdev = best_row['std_test_score']
    best_param = best_row['param_' + param_name]

    display(pd.DataFrame(cv_results) \
            .sort_values(by='rank_test_score').head(num_results))

    scores_df = scores_df.sort_values(by='param_' + param_name)

    if negative:
        means = -scores_df['mean_test_score']
    else:
        means = scores_df['mean_test_score']
    stds = scores_df['std_test_score']
    params = scores_df['param_' + param_name]

    if graph:
        plt.figure(figsize=(8, 8))
        plt.errorbar(params, means, yerr=stds)

        plt.axhline(y=best_mean + best_stdev, color='red')
        plt.axhline(y=best_mean - best_stdev, color='red')
        plt.plot(best_param, best_mean, 'or')

        plt.title(param_name + " vs Score\nBest Score {:0.5f}".format(clf_score))
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.show()


def process_dataset():
    foldernames = os.listdir('bbc')
    foldernames = foldernames

    df_dict = dict.fromkeys(foldernames, 0)
    results = defaultdict(list)
    for folder_no in range(len(foldernames)):
        if foldernames[folder_no] == '.DS_Store' or foldernames[folder_no] == 'README.TXT':
            continue

        my_dir_path = "bbc/" + foldernames[folder_no] + "/"
        for file in Path(my_dir_path).iterdir():
            with open(file, "r", encoding="utf8", errors='ignore') as file_open:
                results["file_name"].append(file.name)
                results["text"].append(file_open.read())
                results["label"].append(foldernames[folder_no])
                results["text"][len(results["text"])-1] = remove_punctuation(results["text"][len(results["text"])-1])

        dataf = pd.DataFrame(results)
    return dataf


def main():

    dataf = process_dataset()

    X_tfid_train = gen_training_set_tfid(dataf['text'])
    X_count_train = gen_training_set_count(dataf['text'])
    X_bigram_train = gen_training_set_count_bigram(dataf['text'])

    X_tfid_train = SelectKBest(chi2, k=500).fit_transform(X_tfid_train,dataf['label'])
    X_count_train = SelectKBest(chi2, k=500).fit_transform(X_count_train,dataf['label'])
    X_bigram_train = SelectKBest(chi2, k=500).fit_transform(X_bigram_train,dataf['label'])


    X_train,X_test,Y_train,Y_test = train_test_split(X_tfid_train , dataf['label'], random_state=4063)

    # params = { 'max_depth' : [3, 4, 5],
    #
    # 'max_features' : [1,2,3,4,5,6,7,8,9],
    #
    # 'min_samples_leaf' : [1,2,3,4,5,6,7,8,9],
    #
    # 'criterion' :["gini", "entropy"]
    # }
    # est = DecisionTreeClassifier()
    # clf = RandomizedSearchCV(est, params, cv=5, verbose=2)
    # g = clf.fit(X_train,Y_train)
    #
    # GridSearch_table_plot(g,'max_depth')
    # GridSearch_table_plot(g,'max_features')
    # GridSearch_table_plot(g,'min_samples_leaf')
    # GridSearch_table_plot(g,'criterion')


    clf = RandomForestClassifier(n_estimators= 800, min_samples_split=5, min_samples_leaf=1,max_features='sqrt', max_depth= 90, bootstrap= False)
    clf__fit = clf.fit(X_train,Y_train)
    clf_preds = clf.predict(X_test)

    neigh = KNeighborsClassifier(leaf_size=41,metric='minowski',n_neighbors=17,p=2,weights='uniform')
    neigh_fit = clf.fit(X_train, Y_train)
    neigh_preds = clf.predict(X_test)

    tree = DecisionTreeClassifier(min_samples_split=2, max_leaf_nodes=90,max_depth=12,criterion='gini')
    tree_fit = tree.fit(X_train,Y_train)
    tree_preds = tree.predict(X_test)

    svc = SVC(random_state=0)
    svc_fit = svc.fit(X_train,Y_train)
    svc_preds = clf.predict(X_test)

    print('----tfid---------------- Random Forest Classifier ----------------------')
    print(classification_report(Y_test, clf_preds))
    print('----tfid-------------- K Nearest Neighbors Classifier ------------------')
    print(classification_report(Y_test, neigh_preds))
    print('----tfid---------------- Decision Tree Classifier ----------------------')
    print(classification_report(Y_test, tree_preds))
    print('----tfid--------------------- SVC Classifier ---------------------------')
    print(classification_report(Y_test, svc_preds))

    X_train,X_test,Y_train,Y_test = train_test_split(X_count_train , dataf['label'], random_state=4063)


    clf = RandomForestClassifier(n_estimators= 800, min_samples_split=5, min_samples_leaf=1,max_features='sqrt', max_depth= 90, bootstrap= False)
    clf__fit = clf.fit(X_train,Y_train)
    clf_preds = clf.predict(X_test)

    neigh = KNeighborsClassifier(leaf_size=41,metric='minowski',n_neighbors=17,p=2,weights='uniform')
    neigh_fit = clf.fit(X_train, Y_train)
    neigh_preds = clf.predict(X_test)

    tree = DecisionTreeClassifier(min_samples_split=2, max_leaf_nodes=90,max_depth=12,criterion='gini')
    tree_fit = tree.fit(X_train,Y_train)
    tree_preds = tree.predict(X_test)

    svc = SVC(random_state=0)
    svc_fit = svc.fit(X_train,Y_train)
    svc_preds = clf.predict(X_test)

    print('-----count--------------- Random Forest Classifier ----------------------')
    print(classification_report(Y_test, clf_preds))
    print('-----count------------- K Nearest Neighbors Classifier ------------------')
    print(classification_report(Y_test, neigh_preds))
    print('-----count--------------- Decision Tree Classifier ----------------------')
    print(classification_report(Y_test, tree_preds))
    print('-----count-------------------- SVC Classifier ---------------------------')
    print(classification_report(Y_test, svc_preds))

    X_train,X_test,Y_train,Y_test = train_test_split(X_bigram_train , dataf['label'], random_state=4063)


    clf = RandomForestClassifier(n_estimators= 800, min_samples_split=5, min_samples_leaf=1,max_features='sqrt', max_depth= 90, bootstrap= False)
    clf__fit = clf.fit(X_train,Y_train)
    clf_preds = clf.predict(X_test)

    neigh = KNeighborsClassifier(leaf_size=41,metric='minowski',n_neighbors=17,p=2,weights='uniform')
    neigh_fit = clf.fit(X_train, Y_train)
    neigh_preds = clf.predict(X_test)

    tree = DecisionTreeClassifier(min_samples_split=2, max_leaf_nodes=90,max_depth=12,criterion='gini')
    tree_fit = tree.fit(X_train,Y_train)
    tree_preds = tree.predict(X_test)

    svc = SVC(random_state=0)
    svc_fit = svc.fit(X_train,Y_train)
    svc_preds = clf.predict(X_test)

    print('-----bigram--------------- Random Forest Classifier ----------------------')
    print(classification_report(Y_test, clf_preds))
    print('-----bigram------------- K Nearest Neighbors Classifier ------------------')
    print(classification_report(Y_test, neigh_preds))
    print('-----bigram--------------- Decision Tree Classifier ----------------------')
    print(classification_report(Y_test, tree_preds))
    print('-----bigram-------------------- SVC Classifier ---------------------------')
    print(classification_report(Y_test, svc_preds))




main()
