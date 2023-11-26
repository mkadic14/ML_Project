import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

def load_data(file_path, delimiter=' ', skiprows=None):
    try:
        data = pd.read_csv(file_path, delimiter=delimiter, header=None, engine='python', skiprows=skiprows)
    except pd.errors.ParserError as e:
        print(f"Error reading file {file_path}: {e}")
        print("Trying with a different parsing strategy...")
        # Handle multiple spaces as delimiter using a raw string for the regex
        data = pd.read_csv(file_path, delimiter=r'\s+', header=None, engine='python', skiprows=skiprows)
    return data

# Paths to the datasets
train_data_path = 'Classification/Dataset4/TrainData4.txt'
test_data_path = 'Classification/Dataset4/TestData4.txt'
train_labels_path = 'Classification/Dataset4/TrainLabel4.txt'

# Load the datasets
train_data_4 = load_data(train_data_path)
test_data_4 = load_data(test_data_path)
train_labels_4 = load_data(train_labels_path)

# Train a Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(train_data_4, train_labels_4.values.ravel())

# Evaluate the model
train_predictions_4 = rf.predict(train_data_4)
print("Classification Report for Training Data (Dataset 4):")
print(classification_report(train_labels_4, train_predictions_4))

# Predict on the test set
test_predictions_4 = rf.predict(test_data_4)

# Export Test Predictions to a CSV file
test_predictions_4_df = pd.DataFrame(test_predictions_4, columns=['Predicted_Label'])
test_predictions_4_file_csv = 'Classification/Dataset4/TestPredictions4.csv'
test_predictions_4_df.to_csv(test_predictions_4_file_csv, index=False)

# Export Test Predictions to a text file without headers
test_predictions_4_file_txt = 'Classification/Dataset4/TestPredictions4.txt'
np.savetxt(test_predictions_4_file_txt, test_predictions_4, fmt='%d', delimiter='\t')

print(f"Test predictions for Dataset 4 exported to {test_predictions_4_file_csv} and {test_predictions_4_file_txt}")