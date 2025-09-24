import numpy as np

class LogisticRegression:
    def __init__(self, alpha : float, iterations : int):
        self.b = 0
        self.m = 0
        self.training_rate = alpha
        self.iterations = iterations
    
    def _gradient_descent(self, x : np.ndarray, y : np.ndarray):
        z = np.dot(x, self.w.T) + self.b
        y_hat = 1 / (1 + np.exp(-z))
        dw = np.dot(x.T, (y_hat - y)) / self.m
        db = np.sum(y_hat - y) / self.m
        return dw, db
    
    def fit(self, x : np.ndarray, y : np.ndarray):
        self.m = x.shape[0]
        self.w = np.zeros(x.shape[1])
        for i in range(self.iterations):
            dw, db = self._gradient_descent(x,y)
            self.w -= self.training_rate * dw
            self.b -= self.training_rate * db
    
    def get_weights(self):
        return self.w, self.b
    



