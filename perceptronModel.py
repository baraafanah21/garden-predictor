
class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.threshold = 0.0
        self.loss_history = []
        self.accuracy_history = []

    def activation(self, value):
        if value >= self.threshold:
            return 1
        return 0

    def weighted_sum(self, x):
        total = 0.0
        for i in range(len(self.weights)):
            total += self.weights[i] * x[i]
        return total

    def predictSingle(self, x):
        return self.activation(self.weighted_sum(x))

    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self.predictSingle(x))
        return predictions

    def calculateAccuracy(self, y_true, y_pred):
        if len(y_true) == 0:
            raise ValueError("Cannot calculate accuracy for an empty dataset.")

        correct = 0
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                correct += 1
        return correct / len(y_true)

    def fit(self, X, y):
        if not X:
            raise ValueError("Training data cannot be empty.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows.")

        num_features = len(X[0])
        self.weights = [0.0] * num_features
        self.threshold = 0.0

        for epoch in range(self.epochs):
            errors = 0

            for i in range(len(X)):
                prediction = self.predictSingle(X[i])
                error = y[i] - prediction

                if error != 0:
                    errors += 1

                for j in range(num_features):
                    self.weights[j] += self.learning_rate * error * X[i][j]

                self.threshold -= self.learning_rate * error

            predictions = self.predict(X)
            accuracy = self.calculateAccuracy(y, predictions)

            self.loss_history.append(errors)
            self.accuracy_history.append(accuracy)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        return self.calculateAccuracy(y, predictions)
