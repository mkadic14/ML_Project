import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import numpy as np

# Load datasets
train_data_2_path = 'Classification/Dataset2/TrainData2.txt'
test_data_2_path = 'Classification/Dataset2/TestData2.txt'
train_labels_2_path = 'Classification/Dataset2/TrainLabel2.txt'

train_data_2 = pd.read_csv(train_data_2_path, delim_whitespace=True, header=None)
test_data_2 = pd.read_csv(test_data_2_path, delim_whitespace=True, header=None)
train_labels_2 = pd.read_csv(train_labels_2_path, header=None)

# Use PCA to reduce the dimensionality
pca = PCA(n_components=0.95)  # Retain 95% of the variance
X_train_pca_2 = pca.fit_transform(train_data_2)
X_test_pca_2 = pca.transform(test_data_2)

# Initialize the Gradient Boosting Classifier
gbc = GradientBoostingClassifier()

# Train the classifier
gbc.fit(X_train_pca_2, train_labels_2.values.ravel())

# Evaluate the model
train_predictions_2 = gbc.predict(X_train_pca_2)
print("Classification Report for Training Data (Dataset 2):")
print(classification_report(train_labels_2, train_predictions_2))

# Predict on the test set
test_predictions_2 = gbc.predict(X_test_pca_2)

# Export Test Predictions to a CSV file
test_predictions_2_df = pd.DataFrame(test_predictions_2, columns=['Predicted_Label'])
test_predictions_2_file_csv = 'Classification/Dataset2/TestPredictions2.csv'
test_predictions_2_df.to_csv(test_predictions_2_file_csv, index=False)

# Export Test Predictions to a text file without headers
test_predictions_2_file_txt = 'Classification/Dataset2/TestPredictions2.txt'
np.savetxt(test_predictions_2_file_txt, test_predictions_2, fmt='%d', delimiter='\t')

print(f"Test predictions for Dataset 2 exported to {test_predictions_2_file_csv} and {test_predictions_2_file_txt}")