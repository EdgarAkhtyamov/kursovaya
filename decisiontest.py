import math

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
import pandas
import ast
import json

with open('dataset.csv', 'r') as dataset:

    data = pandas.read_csv("dataset.csv", sep=';')
    data.head()

    x = data["Matrix"].apply(eval)
    y = data['Res']

    for i in range(len(x)):
        for j in range(len(x[i])):
            for k in range(len(x[i])):
                x[i][j][k] = int(x[i][j][k])


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    clf = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

    vectorizer = CountVectorizer()
    x_train = vectorizer.fit_transform(x_train)

    clf = clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    print("Accuracy on training set: {:.3f}".format(clf.score(x_train, y_train)))
    print("Accuracy on test set: {:.3f}".format(clf.score(x_test, y_test)))

