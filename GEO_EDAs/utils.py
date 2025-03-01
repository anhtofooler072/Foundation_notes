import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import gspread as gs
from gspread_dataframe import set_with_dataframe
from gspread_formatting import CellFormat, Color, set_frozen, set_column_width, format_cell_range


def C_PhiCalulator(csv, plot=False, message=False, flatten=True):
    """
    Calculate the cohesion (c) and friction angle (φ) from a CSV file containing shear resistance data.
    
    Args:
        csv (str): Path to the CSV file containing the data. The CSV should have columns representing normal stress values and rows representing shear resistance values.
        plot (bool, optional): If True, plots the shear stress vs. normal stress data along with the regression line. Default is False.
        message (bool, optional): If True, prints the calculated cohesion and friction angle. Default is False.
    
    Returns:
        tuple: A tuple containing:
            - c (float): The calculated cohesion.
            - phi (float): The calculated friction angle in radians.
    """

    # Read the CSV file
    data = pd.read_csv(csv)

    # Display the first few rows of the dataframe to ensure it's loaded correctly
    # print(data.head())

    if flatten:
        # Flatten the data
        normal_stress = np.repeat(data.columns.astype(float).values, data.shape[0]).reshape(-1, 1)
        shear_resistance = np.concatenate([data[col].values for col in data.columns])
    else:
        # Convert the data to a 1D array
        normal_stress = data.iloc[:,1].astype(float).values.reshape(-1, 1)   
        shear_resistance = data.iloc[:,0].astype(float).values

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
    
def DesignValuesSheetUpdate(authpath: str, clean_data, characteristic_value, rho_ultimate, rho_serviceability, workSheetName: str = None):
    """
    Updates a Google Sheets worksheet with design values and formatted cells.
    Args:
        authpath (str): Path to the Google Sheets API service account credentials file.
        clean_data (DataFrame): DataFrame containing the clean data to be updated in the worksheet.
        characteristic_value (float): The characteristic value to be used in the calculations.
        rho_ultimate (float): The ultimate limit state factor.
        rho_serviceability (float): The serviceability limit state factor.
        workSheetName (str, optional): The name of the worksheet to update. If not provided, a new worksheet will be created.
    Raises:
        gs.WorksheetNotFound: If the specified worksheet is not found and cannot be created.
    Returns:
        None
    """
    
    gc = gs.service_account(filename=authpath)

    ouputcell_format = CellFormat(
        backgroundColor=Color.fromHex('#80b781'),  # Green color
        textFormat={'italic': True, 'fontSize': 14, 'fontFamily': 'Montserrat'}
    )

    Sheet = gc.open('soil_props_pysheet')
    
    try:
        workingSheet = Sheet.worksheet(workSheetName)
    except gs.WorksheetNotFound:
        Sheet.add_worksheet(title=workSheetName, rows=100, cols=100)
        workingSheet = Sheet.worksheet(workSheetName)
    
    workingSheet = Sheet.worksheet(workSheetName)
     
    # Update the sheet with the clean_data dataframe
    if clean_data is not None:
        set_with_dataframe(workingSheet, clean_data)

    if rho_serviceability is not None and rho_ultimate is not None:
        format_cell_range(workingSheet, 'E4:E5', ouputcell_format)
        workingSheet.update('E4:E5', [['gamma I'], ['gamma II']]) 
        workingSheet.update('F4:F5', [[f'{characteristic_value:.2f}(1 ± {rho_ultimate:.4f})'], [f'{characteristic_value:.2f}(1 ± {rho_serviceability:.4f})']])
        print(f"Data updated in the worksheet: {workSheetName}")
    else:
        workingSheet.update_cell(4, 5, 'gamma_C')
        workingSheet.update_cell(5, 5, f'{characteristic_value:.2f}')
        print(f"Data updated in the worksheet: {workSheetName}")


def soil_limitstate_value(dt_count, dt_variance, characteristic_value, t_path):
    """
    This function calculates the ultimate and serviceability limit state design values for soil properties.

    Parameters:
        dt_count (int): The count of data points in the dataset.
        dt_variance (float): The variance of the dataset.
        characteristic_value (float): The characteristic value of the dataset.
        t_path (str): The file path to the CSV file containing t-coefficients.

    Returns:
        tuple: A tuple containing the ultimate limit state design value (rho_ultimate) and the serviceability limit state design value (rho_serviceability).
    """
    
    ## limit state design value:
    t_coef_data = pd.read_csv(t_path)

    n_index = dt_count - 1
    
    if n_index in t_coef_data['n'].values:
        t_1 = t_coef_data[t_coef_data['n'] == n_index]['0.95'].values[0]
        t_2 = t_coef_data[t_coef_data['n'] == n_index]['0.85'].values[0]
    else:
        closest_values = t_coef_data.iloc[(t_coef_data['n'] - n_index).abs().argsort()[:2]]
        X = closest_values['n'].values.reshape(-1, 1)
        
        model_95 = LinearRegression().fit(X, closest_values['0.95'].values)
        t_1 = model_95.predict(np.array([[n_index]]))[0]
        
        model_85 = LinearRegression().fit(X, closest_values['0.85'].values)
        t_2 = model_85.predict(np.array([[n_index]]))[0]

    print('Design values:')
    print('Ultimate limit state design value t (TTGH I): ',t_1)
    print('Serviceability limit state design value t (TTGH II): ',t_2)

    rho_ultimate =  (t_1 - dt_variance) / np.sqrt(n_index)
    rho_serviceability = (t_2 - dt_variance) / np.sqrt(n_index)

    print('rho_ultimate:', rho_ultimate)
    print('rho_serviceability:', rho_serviceability)
    print('---------------------------------------------')

    print(f'gamma_I = {characteristic_value:.2f}(1 ± {rho_ultimate:.4f})')
    print(f'gamma_II = {characteristic_value:.2f}(1 ± {rho_serviceability:.4f})')

    return rho_ultimate, rho_serviceability