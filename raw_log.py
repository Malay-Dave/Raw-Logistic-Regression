import math
import argparse

class LogisticRegression:
    
    def __init__(self, learning_rate=0.01, iterations=1000):
                self.lr = learning_rate

                self.weights = None
                self.bias = 0
                self.iterations = iterations
    def _sigmoid(self, z):
        return 1 / (1 + math.exp(-z))
 
    def fit(self, X, y):
        n_features = len(X[0])

        n_samples = len(X)
        
    
        # Initialize weights to zero
        self.bias = 0.0

        self.weights = [0.0] * n_features
        
        # Gradient Descent
        for _ in range(self.iterations):
            for i in range(n_samples):
                # Calculate dot product: z = (w * x) + b
                linear_model = sum(X[i][j] * self.weights[j] for j in range(n_features)) + self.bias
                y_predicted = self._sigmoid(linear_model)

                # Calculate gradients (error = prediction - actual)
                error = y_predicted - y[i]
                
                
                for j in range(n_features):
                    self.weights[j] -= self.lr * error * X[i][j]
                self.bias -= self.lr * error

    def predict(self, X):
        predictions = []
        for x in X:
            linear_model = sum(x[j] * self.weights[j] for j in range(len(x))) + self.bias
            y_predicted = self._sigmoid(linear_model)
            predictions.append(1 if y_predicted > 0.5 else 0)

        return predictions
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-x1", type=float, required=True, help="Score for Exam 1")
    parser.add_argument("-x2", type=float, required=True, help="Score for Exam 2")
    args = parser.parse_args()
    y_train = [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1]
    X_train = [[6.4, 4.2], [2.2, 8.4], [8.9, 4.5], [0.3, 5.3], [0.3, 5.2], [5.4, 5.3], [8.1, 4.0], [7.0, 6.0], [9.6, 6.0], [1.0, 9.1], [8.1, 8.4], [9.7, 6.3], [8.3, 7.7], [5.8, 8.2], [2.3, 5.7], [2.3, 4.6], [6.4, 6.2], [2.1, 5.6], [6.5, 7.7], [7.3, 5.0], [9.9, 7.8], [6.8, 9.1], [2.3, 4.2], [2.7, 5.3], [8.8, 5.9], [4.0, 9.5], [2.6, 5.5], [2.6, 7.5], [4.0, 5.3], [5.1, 4.5], [1.1, 7.8], [4.2, 4.4], [10.0, 7.2], [8.6, 4.1], [6.8, 7.2], [6.4, 4.7], [4.5, 9.7], [2.6, 7.0], [9.1, 9.2], [6.4, 7.7], [7.6, 7.2], [5.3, 4.0], [0.2, 9.6], [8.3, 5.8], [8.8, 9.7], [4.9, 4.4], [7.7, 4.8], [5.5, 5.6], [4.2, 5.3], [7.3, 5.2]]



    model = LogisticRegression(learning_rate=0.1, iterations=5000)
    model.fit(X_train, y_train)


    test_student = [[args.x1, args.x2]]
    prediction = model.predict(test_student)
    print(f"Bias: {model.bias}")
    print(f"Weights: {model.weights}")
   
    print(f"Prediction for [{args.x1}, {args.x2}]: {'Pass' if prediction[0] == 1 else 'Fail'}")