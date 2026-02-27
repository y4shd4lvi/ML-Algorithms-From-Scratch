import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y):
        mean_x = np.mean(X)
        mean_y = np.mean(y)

        numerator = np.sum((X - mean_x) * (y - mean_y))
        denominator = np.sum((X - mean_x) ** 2)

        self.w = numerator / denominator
        self.b = mean_y - (self.w * mean_x)

    def predict(self, X):
        return self.w * X + self.b