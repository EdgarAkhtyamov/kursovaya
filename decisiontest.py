import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas
import ast

data = pandas.read_csv("dataset.csv", sep=';')

#print(data.head())

x = data["Matrix"].apply(ast.literal_eval)
x = x.apply(lambda row: [list(map(int, sublist)) for sublist in row])

flattened_matrices = [np.array(matrix).flatten() for matrix in x]
X = np.array(flattened_matrices)
y = data['Res']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

clf = DecisionTreeClassifier(criterion="entropy", max_depth=20, min_samples_leaf=1, min_samples_split=2)
clf = clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print("Accuracy on training set: {:.3f}".format(clf.score(x_train, y_train)))
print("Accuracy on test set: {:.3f}".format(clf.score(x_test, y_test)))



