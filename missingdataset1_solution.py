import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_median_imputation(file_path, dataset_name, output_csv_path, output_txt_path):
    # Load the dataset
    data = pd.read_csv(file_path, delimiter='\t', header=None)

    # Define the missing value indicator
    missing_value = 1.00000000000000e+99

    # Replace missing values with NaN
    data.replace(missing_value, np.nan, inplace=True)

    # Keep a copy of the original data
    original_data = data.copy()

    # Perform Median Imputation
    median_imputed_data = original_data.fillna(original_data.median())

    # Comparing the original and imputed datasets
    print(f"Original Dataset Statistics ({dataset_name}):")
    print(original_data.describe())

    print(f"\nMedian Imputed Dataset Statistics ({dataset_name}):")
    print(median_imputed_data.describe())

    # Save the median-imputed dataset to a CSV file without headers
    median_imputed_data.to_csv(output_csv_path, index=False, header=None)

    # Save the median-imputed dataset to a text (txt) file without headers
    np.savetxt(output_txt_path, median_imputed_data.values, delimiter='\t', fmt='%.8f')

    # Plot histograms for the original data
    plt.figure(figsize=(16, 6))
    original_data.plot(kind='hist', bins=50, alpha=0.5, title='Histograms of Original Data')
    plt.legend(loc='upper right')
    plt.show()

    # Plot histograms for the median imputed data
    plt.figure(figsize=(16, 6))
    median_imputed_data.plot(kind='hist', bins=50, alpha=0.5, title='Histograms of Median Imputed Data')
    plt.legend(loc='upper right')
    plt.show()

    return original_data, median_imputed_data

dataset_name = "Missing Dataset 1"
file_path = "Missing Value Estimation/Missing Dataset1/MissingData1.txt"
output_csv_path = "Missing Value Estimation/Missing Dataset1/ImputedDataset1Result.csv"
output_txt_path = "Missing Value Estimation/Missing Dataset1/ImputedDataset1Result.txt"

original_data, median_imputed_data = analyze_median_imputation(file_path, dataset_name, output_csv_path, output_txt_path)