import pandas
from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier  # neural network

import numpy as np

from math import log10

#def majority(model_predictions):


def plurality(model_predictions):

    return ([sum(x) for x in zip(*model_predictions)])


def cross_entropy(p, q):
    cross_entropy_sum = 0

    for i in range(len(p)):
        if (p[i] == 0) or (q[i] == 0):
            pass
        else:
            cross_entropy_sum = cross_entropy_sum - (p[i] * log10(q[i]))

    return cross_entropy_sum


def cross_entropy_total(model_predictions):

    total_cross_entropy_list = []

    for i, vector_i in enumerate(model_predictions):

        total_cross_entropy = 0

        for j, vector_j in enumerate(model_predictions):

            if (i != j):
                total_cross_entropy = total_cross_entropy + cross_entropy(vector_i, vector_j)

        total_cross_entropy_list.append(total_cross_entropy)

    return total_cross_entropy_list

# loading the iris dataset
iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

number_of_decision_trees = 5

number_of_NN = 1 # max 3

number_of_SVM = 4

# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []

models.append(('NB', GaussianNB()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))

# Decsion trees
if number_of_decision_trees > 1:
    for n in range(0, number_of_decision_trees):
        max_depth = int(np.random.uniform(5, 25))
        models.append(('CART_{}'.format(n), DecisionTreeClassifier(max_depth=max_depth)))
else:
    models.append(('CART', DecisionTreeClassifier()))

# SVM
if number_of_SVM > 1:
    for n in range(0, number_of_SVM):
        c_factor = np.random.choice([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])
        kernel_type = np.random.choice(['poly', 'rbf'])
        if kernel_type == 'poly':
            degree = int(np.random.uniform(3, 5))
            models.append(('SVM', SVC(probability=True, C=c_factor, kernel=kernel_type, degree=degree)))
        else:
            models.append(('SVM', SVC(probability=True, C=c_factor, kernel=kernel_type, gamma='auto')))
else:
    models.append(('SVM', SVC(probability=True)))


#Neural Networks
if number_of_NN == 2:
    models.append(('NN_1', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2, 2), random_state=1)))
    models.append(('NN_2', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 3), random_state=1)))
elif number_of_NN == 3:
    models.append(('NN_1', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2, 2), random_state=1)))
    models.append(('NN_2', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 3), random_state=1)))
    models.append(('NN_3', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(4, 4), random_state=1)))

else:
    models.append(('NN', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 3), random_state=1)))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'

# for name, model in models:
#     kfold = model_selection.KFold(n_splits=10, shuffle=True)
#     cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#     print(msg)

for name, model in models:
    model.fit(X_train, y_train)

## Entropy accuracy

# entropy_accuracy_score = 0
#
# for x in range(0, len(X_test)):
#
#     model_predictions = []
#
#     for name, model in models:
#         model_predictions.append(model.predict_proba(X_test[x].reshape(1, -1))[0])
#
#     entropy_result = cross_entropy_total(model_predictions)
#
#     entropy_result_arg = np.argmin(entropy_result)
#
#     entropy_result_prediction = (models[entropy_result_arg][1]).predict(X_test[x].reshape(1, -1))[0]
#
#     print (entropy_result_prediction)
#
#     if (entropy_result_prediction == y_test[x]):
#         entropy_accuracy_score = entropy_accuracy_score + 1
#
# entropy_total_accuracy = entropy_accuracy_score/len(X_test)
#
# print ("Score of Entropy method: ", entropy_total_accuracy)

# Plurality score

plurality_accuracy_score = 0

for x in range(0, len(X_test)):
    model_predictions = []

    for name, model in models:
        model_predictions.append(model.predict_proba(X_test[x].reshape(1, -1))[0])

    plurality_result = plurality(model_predictions)

    plurality_prediction = np.argmax(plurality_result)

    print (plurality_prediction, y_test[x])

    if (plurality_prediction == y_test[x]):
        plurality_accuracy_score = plurality_accuracy_score + 1

plurality_total_accuracy = plurality_accuracy_score/len(X_test)

print ("Score of Plurality method: ", plurality_total_accuracy)








