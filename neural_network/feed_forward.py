from typing import Callable, Tuple, Any, Sequence
import numpy as np
import numpy.typing as npt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier


class Layer:
    def __init__(
            self,
            input_size: int,
            output_size: int,
            activation_function: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(output_size)

    def forward(self, input):
        y = input @ self.weights + self.bias
        return self.activation_function(y)

    def get_parameters(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return self.weights.copy(), self.bias.copy()

    def set_parameters(self, weights: npt.NDArray[np.float64], bias: npt.NDArray[np.float64]):
        self.weights = weights.copy()
        self.bias = bias.copy()


class FeedForwardNetwork:
    def __init__(
            self,
            *,
            layers: list[Layer],
            loss_function: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]],
            metrics: list[Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]]] = None
    ):
        for i in range(len(layers) - 1):
            assert layers[i].output_size == layers[i + 1].input_size
        self.layers = layers
        self.loss_function = loss_function
        self.metrics = metrics if metrics is not None else []

    def forward(self, input: npt.NDArray[np.float64]):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def evaluate(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> Sequence[float]:
        y_pred = self.forward(X)
        loss = np.mean(self.loss_function(y, y_pred))
        metrics = [np.mean(metric(y, y_pred)) for metric in self.metrics]
        return loss, *metrics

    def train_brute_force(
            self,
            X: npt.NDArray[np.float64],
            y: npt.NDArray[np.float64],
            epochs: int
    ) -> list[Sequence[float]]:
        log: list[Sequence[float]] = []
        y_pred = self.forward(X)
        best_loss = np.mean(self.loss_function(y, y_pred))
        best_parameters = self.get_parameters()
        for epoch in range(epochs):
            self._set_random_weights()
            y_pred = self.forward(X)
            loss = np.mean(self.loss_function(y, y_pred))
            if loss < best_loss:
                best_loss = loss
                best_parameters = [layer.get_parameters() for layer in self.layers]
            log.append(self.evaluate(X, y))
        self.set_parameters(best_parameters)
        return log

    def train_with_random_updates(
            self,
            X: npt.NDArray[np.float64],
            y: npt.NDArray[np.float64],
            epochs: int,
            branch_factor: int,
            learning_rate: float
    ) -> list[Sequence[float]]:
        log: list[Sequence[float]] = []
        y_pred = self.forward(X)
        best_loss = np.mean(self.loss_function(y, y_pred))
        best_parameters = self.get_parameters()
        idx = np.arange(X.shape[0])
        for epoch in range(epochs):
            np.random.shuffle(idx)
            initial_parameters = self.get_parameters()
            best_epoch_params = initial_parameters
            best_epoch_loss = np.inf
            for branch in range(branch_factor):
                self.set_parameters(initial_parameters)
                self._update_weights(learning_rate)
                y_pred = self.forward(X)
                loss = np.mean(self.loss_function(y, y_pred))
                if loss < best_epoch_loss:
                    best_epoch_loss = loss
                    best_epoch_params = self.get_parameters()
            if best_epoch_loss < best_loss:
                best_loss = best_epoch_loss
                best_parameters = best_epoch_params
                self.set_parameters(best_parameters)
            else:
                self.set_parameters(initial_parameters)
            log.append(self.evaluate(X, y))
        self.set_parameters(best_parameters)
        return log

    def _set_random_weights(self):
        for layer in self.layers:
            weights = np.random.randn(*layer.weights.shape)
            bias = np.random.randn(*layer.bias.shape)
            layer.set_parameters(weights, bias)

    def _update_weights(self, learning_rate: float):
        for layer in self.layers:
            weights, bias = layer.get_parameters()
            weights += learning_rate * np.random.randn(*weights.shape)
            bias += learning_rate * np.random.randn(*bias.shape)
            layer.set_parameters(weights, bias)

    def get_parameters(self):
        return [layer.get_parameters() for layer in self.layers]

    def set_parameters(self, parameters):
        for layer, parameter in zip(self.layers, parameters):
            layer.set_parameters(*parameter)

def relu(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.maximum(x, 0)


def sigmoid(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 1 / (1 + np.exp(-x))


def squared_error(y: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return (y - np.squeeze(y_pred)) ** 2


def accuracy(y: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return (y == (np.squeeze(y_pred) > 0.5)).astype(np.float64)

def plot_training_log(log: list[Sequence[float]], title: str, labels: list[str]):
    fig, ax = plt.subplots(ncols=1, nrows=len(labels), figsize=(10, 7), sharex='all')
    fig.suptitle(title)
    for i, label in enumerate(labels):
        ax[i].set_ylabel(label)
        ax[i].plot(range(len(log)), [entry[i] for entry in log])
    ax[-1].set_xlabel("Epochs")
    fig.tight_layout()
    fig.savefig(f"{title.lower().replace(' ', '_')}.svg")
    # plt.show()


if __name__ == '__main__':

    # Define network architecture
    layers = [
        Layer(2, 3, relu),
        Layer(3, 4, relu),
        Layer(4, 1, sigmoid)
    ]
    network = FeedForwardNetwork(
        layers=layers,
        loss_function=squared_error,
        metrics=[accuracy]
    )

    # Load wine database, reduce dimensionality to two dimensions and reduce
    # target classes to two classes
    data = load_wine()
    scaled = RobustScaler().fit_transform(data['data'])
    X = PCA(n_components=2).fit_transform(scaled)
    # X = scaled
    y = (data['target'] > 1).astype(np.int8)
    # Split dataset in training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, shuffle=True)

    # Evaluate network on training and test set on the initial random weights
    train_loss, train_acc = network.evaluate(X_train, y_train)
    test_loss, test_acc = network.evaluate(X_test, y_test)
    initial_parameters = network.get_parameters()
    print("Initial Random Weights")
    print(f'Training  loss: {train_loss:.4f}  accuracy: {train_acc:.4f}')
    print(f'Test      loss: {test_loss:.4f}  accuracy: {test_acc:.4f}')

    # Train the network brute-force
    training_log = network.train_brute_force(X_train, y_train, epochs=1000)
    plot_training_log(training_log, "Brute Force", ["Loss", "Accuracy"])
    train_loss, train_acc = network.evaluate(X_train, y_train)
    test_loss, test_acc = network.evaluate(X_test, y_test)
    print("\n\nBrute Force")
    print(f'Training  loss: {train_loss:.4f}  accuracy: {train_acc:.4f}')
    print(f'Test      loss: {test_loss:.4f}  accuracy: {test_acc:.4f}')

    # Reset weights and train the network brute-force, but with small incremental updates in 10 branches, always
    # choosing the locally best branch
    network.set_parameters(initial_parameters)
    training_log = network.train_with_random_updates(X_train, y_train, epochs=1000, branch_factor=10, learning_rate=0.01)
    plot_training_log(training_log, "Random Descent", ["Loss", "Accuracy"])
    train_loss, train_acc = network.evaluate(X_train, y_train)
    test_loss, test_acc = network.evaluate(X_test, y_test)
    print("\n\nRandom Descent")
    print(f'Training  loss: {train_loss:.4f}  accuracy: {train_acc:.4f}')
    print(f'Test      loss: {test_loss:.4f}  accuracy: {test_acc:.4f}')

    # Compare the results to a network trained with gradient descent
    clf = MLPClassifier(hidden_layer_sizes=(3, 4), max_iter=1000)
    clf.fit(X_train, y_train)
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    train_loss, train_acc = np.mean(squared_error(y_train, pred_train)), np.mean(accuracy(y_train, pred_train))
    test_loss, test_acc = np.mean(squared_error(y_test, pred_test)), np.mean(accuracy(y_test, pred_test))
    print("\n\nGradient Descent")
    print(f'Training  loss: {train_loss:.4f}  accuracy: {train_acc:.4f}')
    print(f'Test      loss: {test_loss:.4f}  accuracy: {test_acc:.4f}')