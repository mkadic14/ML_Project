import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def load_data(file_path, delimiter=None):
    # If delimiter is not specified, default to None, which works for newline-separated data
    data = pd.read_csv(file_path, delimiter=delimiter, header=None)
    return data

# Paths to the datasets
train_data_path = 'Classification/Dataset3/TrainData3.txt'
test_data_path = 'Classification/Dataset3/TestData3.txt'
train_labels_path = 'Classification/Dataset3/TrainLabel3.txt'

# Load the datasets
train_data_3 = load_data(train_data_path, delimiter='\t')
test_data_3 = load_data(test_data_path, delimiter=',')
train_labels_3 = load_data(train_labels_path)  # Newline-separated

# Cap Large Values
cap_value = np.finfo(np.float32).max
train_data_3[train_data_3 > cap_value] = cap_value
test_data_3[test_data_3 > cap_value] = cap_value

# Handle missing data with Mean Imputation
imputer = SimpleImputer(strategy='mean')
train_data_3_imputed = imputer.fit_transform(train_data_3)
test_data_3_imputed = imputer.transform(test_data_3)

# Convert to float64
train_data_3_imputed = train_data_3_imputed.astype(np.float64)
test_data_3_imputed = test_data_3_imputed.astype(np.float64)

# Train a Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(train_data_3_imputed, train_labels_3.values.ravel())

# Evaluate the model
train_predictions_3 = rf.predict(train_data_3_imputed)
print("Classification Report for Training Data (Dataset 3):")
print(classification_report(train_labels_3, train_predictions_3))

# Predict on the test set
test_predictions_3 = rf.predict(test_data_3_imputed)

# Export Test Predictions to a CSV file
test_predictions_3_df = pd.DataFrame(test_predictions_3, columns=['Predicted_Label'])
test_predictions_3_file_csv = 'Classification/Dataset3/TestPredictions3.csv'
test_predictions_3_df.to_csv(test_predictions_3_file_csv, index=False)

# Export Test Predictions to a text file without headers
test_predictions_3_file_txt = 'Classification/Dataset3/TestPredictions3.txt'
np.savetxt(test_predictions_3_file_txt, test_predictions_3, fmt='%d', delimiter='\t')

print(f"Test predictions for Dataset 3 exported to {test_predictions_3_file_csv} and {test_predictions_3_file_txt}")