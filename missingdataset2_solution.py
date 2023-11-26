import pandas as pd
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import numpy as np

def analyze_knn_imputation(file_path, dataset_name, output_csv_path, output_txt_path, n_neighbors=5):
    # Load the dataset
    data = pd.read_csv(file_path, delimiter='\t', header=None)

    # Define the missing value indicator
    missing_value = 1.00000000000000e+99

    # Replace missing values with NaN
    data.replace(missing_value, np.nan, inplace=True)

    # Initialize the KNN Imputer
    imputer = KNNImputer(n_neighbors=n_neighbors)

    # Perform KNN Imputation
    imputed_data = imputer.fit_transform(data)

    # Create a DataFrame from the imputed data
    imputed_data_df = pd.DataFrame(imputed_data, columns=data.columns)

    # Compare the original and imputed datasets
    print(f"Original Dataset Statistics ({dataset_name}):")
    print(data.describe())

    print(f"\nKNN Imputed Dataset Statistics ({dataset_name}):")
    print(imputed_data_df.describe())

    # Save the KNN imputed dataset to a CSV file without headers
    imputed_data_df.to_csv(output_csv_path, index=False, header=None)

    # Save the KNN imputed dataset to a text (txt) file without headers
    np.savetxt(output_txt_path, imputed_data, delimiter='\t', fmt='%.8f')

    # Plot histograms for the original data
    plt.figure(figsize=(16, 6))
    plt.title('Histograms of Original Data')
    for i in data.columns:
        plt.hist(data[i].dropna(), bins=50, alpha=0.5, label=str(i))
    plt.legend(loc='upper right')
    plt.show()

    # Plot histograms for the KNN imputed data
    plt.figure(figsize=(16, 6))
    plt.title('Histograms of KNN Imputed Data')
    for i in imputed_data_df.columns:
        plt.hist(imputed_data_df[i], bins=50, alpha=0.5, label=str(i))
    plt.legend(loc='upper right')
    plt.show()

    return data, imputed_data_df

dataset_name = "Dataset 2"
file_path = "Missing Value Estimation/Missing Dataset2/MissingData2.txt"
output_csv_path = "Missing Value Estimation/Missing Dataset2/ImputedDataset2Result.csv"
output_txt_path = "Missing Value Estimation/Missing Dataset2/ImputedDataset2Result.txt"

original_data, knn_imputed_data = analyze_knn_imputation(file_path, dataset_name, output_csv_path, output_txt_path)