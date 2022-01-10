from pyts.classification import BOSSVS
import pandas as pd
from sklearn.model_selection import train_test_split
from pyts.classification import TimeSeriesForest


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
    path = '/Users/wenxu/PycharmProjects/DataChallenge/data/age/20<Age<40.csv'
    X, y = prepare_data(path)
    X_train, X_test, y_train, y_test = split(X, y)
    clf = BOSSVS(window_size=28)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(score)

    clf = TimeSeriesForest(random_state=43)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(score)
