from pyts.classification import BOSSVS
import pandas as pd
from sklearn.model_selection import train_test_split
from pyts.classification import TimeSeriesForest
from sklearn.metrics import classification_report


def prepare_data(path):
    df = pd.read_csv(path, sep=',')
    df = df.fillna(0)
    X = df[df.columns[:-1]].to_numpy()
    y = df['SepsisLabel'].to_numpy()
    return X, y


def split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    path = '/Users/wenxu/PycharmProjects/DataChallenge/data/imputed_subgroup/40<linear<70.csv'
    X, y = prepare_data(path)
    X_train, X_test, y_train, y_test = split(X, y)
    clf = BOSSVS(window_size=28)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(score)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    clf = TimeSeriesForest(random_state=43)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(score)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
