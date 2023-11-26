import warnings
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning

def mice_imputation(file_path, output_csv_path, output_txt_path, n_imputations=5, max_iter=10, random_state=0):
    # Load the dataset
    data = pd.read_csv(file_path, delimiter='\t', header=None)

    # Replace the missing value indicator with NaN
    data.replace(1.00000000000000e+99, np.nan, inplace=True)

    # List to hold the multiple imputations
    multiple_imputations = []

    # Perform multiple imputations
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for i in range(n_imputations):
            imputer = IterativeImputer(max_iter=max_iter,
                                   random_state=random_state + i)  # Different seed for each imputation
            imputed_data = imputer.fit_transform(data)
            multiple_imputations.append(imputed_data)

    # Combine the multiple imputations to create a final imputed dataset
    final_imputed_data = np.mean(multiple_imputations, axis=0)

    # Convert the imputed data to a DataFrame
    imputed_data_df = pd.DataFrame(final_imputed_data, columns=data.columns)

    # Save the imputed data to a CSV file without headers
    imputed_data_df.to_csv(output_csv_path, index=False, header=False)

    # Save the imputed data to a text file without headers
    np.savetxt(output_txt_path, final_imputed_data, delimiter='\t')

    # Plot histograms for the original data
    plt.figure(figsize=(16, 6))
    plt.title('Histograms of Original Data')
    for i in data.columns:
        # Drop NA to avoid issues with plotting missing values
        plt.hist(data[i].dropna(), bins=50, alpha=0.5, label=f"Column {i}")
    plt.show()

    # Plot histograms for the imputed data
    plt.figure(figsize=(16, 6))
    plt.title('Histograms of Imputed Data')
    for i in imputed_data_df.columns:
        plt.hist(imputed_data_df[i], bins=50, alpha=0.5, label=f"Column {i}")
    plt.show()

    return data, imputed_data_df


file_path = "Missing Value Estimation/Missing Dataset3/MissingData3.txt"
output_csv_path = "Missing Value Estimation/Missing Dataset3/ImputedDataset3Result.csv"
output_txt_path = "Missing Value Estimation/Missing Dataset3/ImputedDataset3Result.txt"
mice_imputed_data = mice_imputation(file_path, output_csv_path, output_txt_path, n_imputations=5, max_iter=10, random_state=0)