import pandas as pd

def analyze_missing_data(file_path, dataset_name):
    # Using space as the delimiter
    data = pd.read_csv(file_path, delim_whitespace=True, header=None)
    missing_value = 1.00000000000000e+99

    # Replace the missing value indicator with NaN
    data = data.replace(missing_value, pd.NA)

    # Calculate the percentage of missing data in each column
    percent_missing_per_column = data.isnull().sum() * 100 / len(data)
    missing_value_df_per_column = pd.DataFrame({'column_name': data.columns,
                                                'percent_missing': percent_missing_per_column})
    print(f"Missing Data Analysis for {dataset_name}:")
    print(missing_value_df_per_column)

    total_cells = data.size  # Total number of values in the dataset
    total_missing = data.isnull().sum().sum()  # Count of missing values
    percent_missing_total = (total_missing / total_cells) * 100
    print(f"Total percentage of missing data in {dataset_name}: {percent_missing_total:.2f}%")

    print(f"\nSummary statistics with missing data for {dataset_name}:")
    print(data.describe())
    print("\n")

# Paths to the datasets
train_data_path = 'Classification/Dataset2/TrainData2.txt'
test_data_path = 'Classification/Dataset2/TestData2.txt'

# Analyze missing data for training dataset
analyze_missing_data(train_data_path, "Training Dataset")

# Analyze missing data for test dataset
analyze_missing_data(test_data_path, "Test Dataset")