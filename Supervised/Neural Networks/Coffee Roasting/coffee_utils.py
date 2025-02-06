import pandas as pd
import numpy as np

data = pd.read_excel('coffee_dataset_binary.xlsx')

def load_data():
    # x_train = data[['Temperature (Celsius)', 'Duration (minutes)']].values
    x_train = np.array([data['Temperature (Celsius)'], data['Duration (minutes)']]).T
    y_train = np.array(data['Label']).reshape(-1,1)
    return x_train, y_train
