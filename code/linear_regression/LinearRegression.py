import numpy as np

class LinearRegression():
    def __init__(self, lr = 0.01, epochs = 100):
        self.lr = lr 
        self.epochs = epochs 
        self.weights = None
        self.bias = 0
    
    def fit(X,y):
        n_samples, n_features = X.shape # get rows and cols 
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):

            y_pred = np.dot(X, self.weights) + self.bias
            difference = y - y_pred
            dw = (1/n_samples) * np.dot(X.T, difference)
            db = (1/n_samples) * np.sum(difference)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):

        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
