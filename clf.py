from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn import metrics
from imblearn.under_sampling import ClusterCentroids
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier


def prepare_data(path):
    df = pd.read_csv(path, sep=',')
    df = df.fillna(0)
    X = df[df.columns[:-1]].to_numpy()
    y = df['SepsisLabel'].to_numpy()
    print('Original dataset shape %s' % Counter(y))
    return X, y


def split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def svc(X_train, X_test, y_train):
    clf = SVC(kernel='sigmoid', degree=10, gamma='auto')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred


def dc(X_train, X_test, y_train):
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred


def report(y_pred, y_test):
    class_names = ['healthy', 'sick']
    print(classification_report(y_test, y_pred, target_names=class_names))
    # report_dict = classification_report(y_test, y_pred, output_dict=True)
    # repdf = pd.DataFrame(report_dict).round(2).transpose()
    # repdf.insert(loc=0, column='class', value=class_names + ["accuracy", "macro avg", "weighted avg"])
    # save_path = '/Users/wenxu/PycharmProjects/DataChallenge/'
    # repdf.to_csv(save_path + "imputed_20<mean<40.csv", index=False)
    print(metrics.roc_auc_score(y_test, y_pred))


def undersampling(X, y):
    cc = ClusterCentroids(random_state=42)
    under_X_res, under_y_res = cc.fit_resample(X, y)
    print('Undersampling dataset shape %s' % Counter(under_y_res))
    return under_X_res, under_y_res


def oversampling(X, y):
    sm = SMOTE(random_state=42)
    over_X_res, over_y_res = sm.fit_resample(X, y)
    print('Oversampling dataset shape %s' % Counter(over_y_res))
    return over_X_res, over_y_res


if __name__ == '__main__':
    path = '/Users/wenxu/PycharmProjects/DataChallenge/data/age/20<Age<40.csv'
    X, y = prepare_data(path)
    X_train, X_test, y_train, y_test = split(X, y)
    y_pred = dc(X_train, X_test, y_train)
    report(y_pred, y_test)

    under_X_res, under_y_res = undersampling(X, y)
    X_train, X_test, y_train, y_test = split(under_X_res, under_y_res)
    y_pred = dc(X_train, X_test, y_train)
    report(y_pred, y_test)

    over_X_res, over_y_res = oversampling(X, y)
    X_train, X_test, y_train, y_test = split(over_X_res, over_y_res)
    y_pred = dc(X_train, X_test, y_train)
    report(y_pred, y_test)
