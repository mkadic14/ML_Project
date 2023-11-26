import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

def load_data(file_path, delimiter=r'\s+'):
    try:
        data = pd.read_csv(file_path, delimiter=delimiter, header=None)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        data = pd.DataFrame()  # Return an empty DataFrame in case of error
    return data

# Load the datasets
train_data_path = 'Classification/Dataset5/TrainData5.txt'
test_data_path = 'Classification/Dataset5/TestData5.txt'
train_labels_path = 'Classification/Dataset5/TrainLabel5.txt'

train_data = load_data(train_data_path)
test_data = load_data(test_data_path)
train_labels = pd.read_csv(train_labels_path, header=None)

# Define the hyperparameters for Random Forest
best_params = {
    'n_estimators': 500,  # Number of trees in the forest
    'max_depth': 50,  # Maximum depth of the tree
    'min_samples_split': 2,  # Minimum number of samples required to split an internal node
    'min_samples_leaf': 1,  # Minimum number of samples required to be at a leaf node
    'bootstrap': True  # Whether bootstrap samples are used when building trees
}

# Initialize the Random Forest classifier with the best hyperparameters
best_rf = RandomForestClassifier(**best_params)

# Train the classifier with the best hyperparameters
best_rf.fit(train_data, train_labels.values.ravel())

# Evaluate the model
train_predictions = best_rf.predict(train_data)
print("Classification Report for Training Data (Dataset 5):")
print(classification_report(train_labels, train_predictions))

# Predict on the test set
test_predictions = best_rf.predict(test_data)

# Export Test Predictions to a CSV file
test_predictions_df = pd.DataFrame(test_predictions, columns=['Predicted_Label'])
test_predictions_file_csv = 'Classification/Dataset5/TestPredictions5.csv'
test_predictions_df.to_csv(test_predictions_file_csv, index=False)

# Export Test Predictions to a text file without headers
test_predictions_file_txt = 'Classification/Dataset5/TestPredictions5.txt'
np.savetxt(test_predictions_file_txt, test_predictions, fmt='%d', delimiter='\t')

print(f"Test predictions for Dataset 5 exported to {test_predictions_file_csv} and {test_predictions_file_txt}")