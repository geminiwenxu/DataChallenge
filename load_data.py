import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


def gender(male, female):
    ls_labels = ["male ", "female"]
    share = [male, female]
    figureObject, axesObject = plt.subplots()
    axesObject.pie(share, labels=ls_labels, autopct='%1.2f', startangle=120)
    axesObject.axis('equal')
    plt.title("class distribution")
    return figureObject


def data_ratio(counter_1, counter_2, counter_3, counter_4):
    ls_labels = ["age<20 ", "20<age<40", "40<age<70", "70<age"]
    share = [counter_1, counter_2, counter_3, counter_4]
    figureObject, axesObject = plt.subplots()
    axesObject.pie(share, labels=ls_labels, autopct='%1.2f', startangle=120)
    axesObject.axis('equal')
    plt.title("class distribution")
    return figureObject


def over_view():
    df = pd.read_csv("/Users/wenxu/PycharmProjects/DataChallenge/data/training_A", sep=',')
    features = list(df.columns)  # check nan per columns

    counts = []
    for feature in features:
        count_nan = df[feature].isnull().sum()
        counts.append(count_nan)
    # plotting into histogram
    fig, ax = plt.subplots()
    x_ax = np.arange(len(features))
    ax.bar(x_ax, counts)
    plt.xticks(x_ax, features, rotation='vertical')
    # adding description to the histogram
    plt.title("Amount of missing value per feature")
    plt.ylabel("Count")
    plt.xlabel("Feature")
    st.pyplot(fig)

    # check nan per row: min/max number of features
    ls_count = []
    for index, row in df.iterrows():
        count_nan = row.isnull().sum().item()
        ls_count.append(count_nan)
    low = 41 - max(ls_count)
    high = 41 - min(ls_count)
    print("min of features:{} max of features: {}".format(41 - max(ls_count), 41 - min(ls_count)))
    st.metric(label="min of features", value=low)
    st.metric(label="max of features", value=high)

    counter_1 = 0
    counter_2 = 0
    counter_3 = 0
    counter_4 = 0
    for index, row in df.iterrows():
        if row['Age'] < 20:
            counter_1 += 1
        elif 20 < row['Age'] < 40:
            counter_2 += 1
        elif 40 < row['Age'] < 70:
            counter_3 += 1
        else:
            counter_4 += 1
    fig = data_ratio(counter_1, counter_2, counter_3, counter_4)
    st.pyplot(fig)

    female = 0
    male = 0
    for index, row in df.iterrows():
        if row['Gender'] == 1:
            male += 1
        else:
            female += 1
    fig_2 = gender(male, female)
    st.pyplot(fig_2)
