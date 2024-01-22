import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Set a random seed
RANDOM_SEED = 1212

# Load preprocessed training and testing data
train_data = pd.read_csv('./data/processed_train.csv')
test_data = pd.read_csv('./data/processed_test.csv')
test_original = pd.read_csv('./data/test.csv')

# Separate features and target variable in the training set
X_train = train_data.iloc[:, 1:]
y_train = train_data['Survived'].values

# Initialize SVM classifier with a slightly different regularization parameter
# Adjust the C parameter to change the regularization strength
svm_classifier = SVC(
    C=1.1,        
    kernel='poly',
    degree=2,
    gamma='auto'
)

# Train the SVM classifier
svm_classifier.fit(X_train, y_train)

# Print accuracy on the training set
train_accuracy = svm_classifier.score(X_train, y_train)
print(f"Accuracy on the training set: {train_accuracy}")

# Make predictions on the test set
predictions = svm_classifier.predict(test_data)

# Create a submission DataFrame
submission_df = pd.DataFrame({
    'PassengerId': test_original['PassengerId'],
    'Survived': predictions
})

# Save predictions to a submission file
submission_df.to_csv('./submission/kernel_method_modified.csv', index=False)
