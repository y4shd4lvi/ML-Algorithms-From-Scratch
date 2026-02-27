from models.simple_linear_regression import LinearRegression
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "data", "simple_linear_regression.csv")

df = pd.read_csv(data_path)