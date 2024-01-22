import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# Set a random seed
RANDOM_SEED = 1212

# Load preprocessed training and testing data
train_data = pd.read_csv('./data/processed_train.csv')
test_data = pd.read_csv('./data/processed_test.csv')
test_original = pd.read_csv('./data/test.csv')

# Separate features and target variable in the training set
X_train = train_data.iloc[:, 1:]
y_train = train_data['Survived'].values

# Model Params - Slightly adjusted hyperparameters
neural_net_classifier = MLPClassifier(
    solver='adam',
    learning_rate_init=0.0025,  # Adjusted learning rate
    learning_rate='constant',
    hidden_layer_sizes=(50, 100),  # Adjusted hidden layer sizes
    alpha=0.00015,  # Adjusted regularization strength
    activation='logistic'
)

# Train the neural network classifier
neural_net_classifier.fit(X_train, y_train)

# Print accuracy on the training set
train_accuracy = neural_net_classifier.score(X_train, y_train)
print(f"Accuracy on the training set: {train_accuracy}")

# Make predictions on the test set
predictions = neural_net_classifier.predict(test_data)

# Create a submission DataFrame
submission_df = pd.DataFrame({
    'PassengerId': test_original['PassengerId'],
    'Survived': predictions
})

# Save predictions to a submission file
submission_df.to_csv('./submission/neural_network_modified.csv', index=False)
