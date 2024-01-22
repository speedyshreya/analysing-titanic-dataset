import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the training and testing data
train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

# Display training and testing data
print(train_data.columns)
print(test_data.columns)

# Check for missing values 
print("Number of NaNs in each column in the training set:")
print(train_data.isnull().sum())
print("Number of NaNs in each column in the testing set:")
print(test_data.isnull().sum())

# Drop redundant columns
train_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

combined_data = pd.concat([train_data, test_data]).reset_index(drop=True)

# Replace NaNs in the training set
train_data['Age'].fillna(combined_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna('S', inplace=True)

# Replace NaNs in the testing set
test_data['Age'].fillna(combined_data['Age'].median(), inplace=True)
test_data['Fare'].fillna(combined_data['Fare'].median(), inplace=True)

# Drop the 'Parch' column
train_data.drop(columns=['Parch'], axis=1, inplace=True)
test_data.drop(columns=['Parch'], axis=1, inplace=True)

# Remove outliers
cols = ['Age', 'SibSp', 'Fare']
train_data[cols] = train_data[cols].clip(lower=train_data[cols].quantile(0.15), upper=train_data[cols].quantile(0.85), axis=1)
train_data.plot(kind='box', figsize=(10, 8), color='lightblue')
plt.savefig('./graph/data_visualization/fixed_outliers_modified.png')

#Changing back to int values
train_data = pd.get_dummies(train_data, columns=['Pclass', 'Sex', 'Embarked'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Pclass', 'Sex', 'Embarked'], drop_first=True)

# Update train and test sets and upload as CSV files
train_data.to_csv('./data/processed_train_modified.csv', index=False)
test_data.to_csv('./data/processed_test_modified.csv', index=False)
