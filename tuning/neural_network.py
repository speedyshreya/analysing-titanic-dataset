import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV


SEED = 1212

train = pd.read_csv('./data/processed_train.csv')
test = pd.read_csv('./data/processed_test.csv')

X_train=train.iloc[:,1:]
y_train=train['Survived'].values
X_test = test

# Number of layers
hidden_layer_sizes = [(x, y) for x in [50, 100, 150, 200] for y in [50, 100, 150, 200]]
# Activation function for the hidden layers
activation = ['identity', 'logistic', 'tanh', 'relu']
# Solver for weight optimization
solver = ['lbfgs', 'sgd', 'adam']
# L2 regularization term
alpha = [0.00001, 0.0001, 0.001, 0.01, 0.1]
# Learning rate
learning_rate = ['constant', 'invsclaing', 'adaptive']
# Learning rate initial value
learning_rate_init = [0.0001, 0.001, 0.01, 0.1]


random_grid = {'hidden_layer_sizes': hidden_layer_sizes,
               'activation': activation,
               'solver': solver,
               'alpha': alpha,
               'learning_rate': learning_rate,
               'learning_rate_init': learning_rate_init}

clf = MLPClassifier()
clf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=SEED, n_jobs = -1)
clf_random.fit(X_train, y_train)
print(clf_random.best_params_)
# {'solver': 'adam', 'learning_rate_init': 0.01, 'learning_rate': 'constant', 'hidden_layer_sizes': (50, 150), 'alpha': 0.0001, 'activation': 'logistic'}