import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def C_PhiCalulator(csv):
    # Read the CSV file
    data = pd.read_csv(csv)

    # Display the first few rows of the dataframe to ensure it's loaded correctly
    print(data.head())

    # Flatten the data
    normal_stress = np.repeat(data.columns.astype(float).values, data.shape[0]).reshape(-1, 1)
    shear_resistance = np.concatenate([data[col].values for col in data.columns])

    # Perform linear regression
    reg = LinearRegression().fit(normal_stress, shear_resistance)
    c = reg.intercept_  # Cohesion
    tan_phi = reg.coef_[0]  # Slope, tan(φ)
    phi = np.arctan(tan_phi)  # Friction angle

    print(f"Cohesion (c): {np.round(c,3)}")
    print(f"Friction angle (φ): {np.round(np.degrees(phi),3)}")

    # Plot the data and the regression line
    plt.scatter(normal_stress, shear_resistance, color='blue')
    plt.plot(normal_stress, reg.predict(normal_stress), color='red')
    plt.xlabel('Normal Stress')
    plt.ylabel('Shear Stress')
    plt.title('Shear Stress vs. Normal Stress')
    plt.show()

    return c, phi