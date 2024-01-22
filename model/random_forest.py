import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


SEED = 1212

# Input Data
train = pd.read_csv('./data/processed_train.csv')
test = pd.read_csv('./data/processed_test.csv')
test_orig = pd.read_csv('./data/test.csv')

X_train=train.iloc[:,1:]
y_train=train['Survived'].values
X_test = test

# Model Params
clf = RandomForestClassifier(
    n_estimators=800,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features=None,
    max_depth=8,
    criterion='entropy'
)

clf.fit(X_train,y_train)

print(clf.score(X_train, y_train))
# Accuracy on training set:
# 0.8731762065095399

predictions = clf.predict(X_test)

# Generate submission file
submission= pd.DataFrame({'PassengerId' : test_orig['PassengerId'], 'Survived': predictions })
submission.to_csv('./submission/random_forest.csv', index=False)
