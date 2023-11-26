import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load datasets
train_data_path = 'Classification/Dataset1/TrainData1.txt'
test_data_path = 'Classification/Dataset1/TestData1.txt'
train_labels_path = 'Classification/Dataset1/TrainLabel1.txt'

train_data = pd.read_csv(train_data_path, delimiter='\t', header=None)
test_data = pd.read_csv(test_data_path, delimiter='\t', header=None)
train_labels = pd.read_csv(train_labels_path, header=None)

# Replace the missing value indicator with NaN
missing_value_indicator = 1.00000000000000e+99
train_data.replace(missing_value_indicator, np.nan, inplace=True)
test_data.replace(missing_value_indicator, np.nan, inplace=True)

# Mean Imputation
mean_imputer = SimpleImputer(strategy='mean')
train_data_imputed = mean_imputer.fit_transform(train_data)
test_data_imputed = mean_imputer.transform(test_data)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_data_imputed)
X_test_scaled = scaler.transform(test_data_imputed)

# Determine the number of components for Incremental PCA
pca = PCA(0.95)
pca.fit(X_train_scaled)
n_components = pca.n_components_
print(f"Number of components to retain 95% variance: {n_components}")

# Apply Incremental PCA
ipca = IncrementalPCA(n_components=n_components)
X_train_pca = ipca.fit_transform(X_train_scaled)
X_test_pca = ipca.transform(X_test_scaled)

# Train the SVM Classifier
svm = SVC(kernel='linear')
svm.fit(X_train_pca, train_labels.values.ravel())

# Evaluate the model
train_predictions = svm.predict(X_train_pca)
print("Classification Report for Training Data:")
print(classification_report(train_labels, train_predictions))

# Predict on the test set
test_predictions = svm.predict(X_test_pca)

# Export Test Predictions to a CSV file
test_predictions_df = pd.DataFrame(test_predictions, columns=['Predicted_Label'])
test_predictions_file_csv = 'Classification/Dataset1/TestPredictions1.csv'
test_predictions_df.to_csv(test_predictions_file_csv, index=False)

# Export Test Predictions to a text file without headers
test_predictions_file_txt = 'Classification/Dataset1/TestPredictions1.txt'
np.savetxt(test_predictions_file_txt, test_predictions, fmt='%d', delimiter='\t')

print(f"Test predictions for Dataset 1 exported to {test_predictions_file_csv} and {test_predictions_file_txt}")