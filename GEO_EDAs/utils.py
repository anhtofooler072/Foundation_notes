import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def C_PhiCalulator(csv, plot=False, message=False):
    # Read the CSV file
    data = pd.read_csv(csv)

    # Display the first few rows of the dataframe to ensure it's loaded correctly
    # print(data.head())

    # Flatten the data
    normal_stress = np.repeat(data.columns.astype(float).values, data.shape[0]).reshape(-1, 1)
    shear_resistance = np.concatenate([data[col].values for col in data.columns])

    # Perform linear regression
    reg = LinearRegression().fit(normal_stress, shear_resistance)
    c = reg.intercept_  # Cohesion
    tan_phi = reg.coef_[0]  # Slope, tan(φ)
    phi = np.arctan(tan_phi)  # Friction angle

    if message:
        print(f"Cohesion (c): {np.round(c,3)}")
        print(f"Friction angle (φ): {np.round(np.degrees(phi),3)} \n")

    #Plot the data and the regression line
    if plot:
        plt.scatter(normal_stress, shear_resistance, color='blue')
        plt.plot(normal_stress, reg.predict(normal_stress), color='red')
        plt.xlabel('Normal Stress')
        plt.ylabel('Shear Stress')
        plt.title('Shear Stress vs. Normal Stress')
        plt.show()

    return c, phi

def read_csv_files(folder_path):
    """
    Reads all .csv files in a given folder and returns a dictionary 
    where keys are filenames and values are pandas DataFrames.

    Args:
        folder_path (str): The path to the folder containing the .csv files.

    Returns:
        dict: A dictionary of DataFrames, or None if an error occurs.
    """
    try:
        dataframes = {}
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(folder_path, filename)
                try:
                    df = pd.read_csv(file_path)
                    dataframes[filename] = df
                except pd.errors.ParserError as e:
                    print(f"Error parsing {filename}: {e}")
                except FileNotFoundError:
                    print(f"File not found: {file_path}")
                except Exception as e:
                    print(f"An unexpected error occurred while processing {filename}: {e}")

        return dataframes
    except FileNotFoundError:
        print(f"Folder not found: {folder_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None