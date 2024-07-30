import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from typing import Tuple


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # Load the Heart Disease dataset
    data = pd.read_csv('heart.csv')
    X = data.drop(columns=['target'], axis=1)
    y = data['target']

    # Perform a train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # Print the shapes of the train and test sets
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    
    return X_train, X_test, y_train, y_test


class NeuralNetwork():
    
    def __init__(self, layer_dimensions : list) -> None:
        
        self.num_layers = len(layer_dimensions)
        self.layers = layer_dimensions
        self.weights = {}
        self.biases = {}
        
        self.initialize_parameters()
    
    
    def fit(self, X : pd.DataFrame, y : np.array) -> None:
        
        self.X = X.values.T
        self.y = y.values


    def train(self, epochs : int = 100, learning_rate : float = 0.01, plot : bool = False) -> list:
            
        costs = []
        
        for epoch in range(1, epochs + 1):
            
            AL, caches = self.forward_propagation()

            # Calculate cost
            cost = self.compute_cost(AL)
            costs.append(cost)
            
            if epoch % 10 == 0:
                print("Cost after epoch %i: %f" %(epoch, cost))
        
        if plot:
            self.plot_costs(costs, learning_rate)
        
        
        return costs


    def plot_costs(self, costs : list, learning_rate : float) -> None:
        
        #plot costs
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    ########################################################################################################################################
    ###################################################### Forward Pass #################################################################### 
    ########################################################################################################################################
    def initialize_parameters(self) -> None:
        
        for l in range(1, self.num_layers):
            
            self.weights["W" + str(l)] = np.random.randn(self.layers[l], self.layers[l-1]) * 0.01
            self.biases["b" + str(l)] = np.random.randn(self.layers[l], 1) * 0.00001 #np.zeros((self.layers[l], 1)) # 
            
            assert(self.weights["W" + str(l)].shape == (self.layers[l], self.layers[l-1]))
            assert(self.biases["b" + str(l)].shape == (self.layers[l], 1))
    
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    
    def relu(self, x):
        return np.maximum(0,x)
    
    
    def linear_step(self, A_prev, W, b):
        
        return np.dot(W, A_prev) + b, (A_prev, W, b)

    
    def linear_activation_function(self, A_prev, W, b, activation):
        
        Z, linear_cache = self.linear_step(A_prev, W, b)
        
        if activation == "sigmoid":
            A = self.sigmoid(Z)
            
        elif activation == "relu":
            A = self.relu(Z)

        return A, (linear_cache, Z)
        
    
    def forward_propagation(self) -> None:
        
        A = self.X
        caches = []

        for l in range(1, self.num_layers - 1):

            A_prev = A
            A, cache = self.linear_activation_function(A_prev, self.weights["W" + str(l)], self.biases["b" + str(l)], activation = "relu")
            caches.append(cache)
        
        AL, cache = self.linear_activation_function(A, self.weights["W" + str(self.num_layers - 1)], self.biases["b" + str(self.num_layers - 1)], activation = "sigmoid")
        caches.append(cache)
        
        return AL, caches
    
    ########################################################################################################################################
    ###################################################### Compute Costs ################################################################### 
    ########################################################################################################################################
    def compute_cost(self, AL) -> float:
        
        m = self.y.shape[0]
        cost = (-1/m) * np.sum(np.multiply(self.y, np.log(AL)) + np.multiply(1 - self.y, np.log(1 - AL)))
        
        cost = np.squeeze(cost)
        
        assert(cost.shape == ())
        
        return cost



############################################################################################################
############################################## Example Usage ###############################################
############################################################################################################
if __name__ == "__main__":

    if len(sys.argv) > 1:
        number_of_iterations = int(sys.argv[1])
    else:
        number_of_iterations = 10
                
    overall_costs = {}
    X_train, X_test, y_train, y_test = load_data()

    for i in range(0, number_of_iterations):
        
        nn = NeuralNetwork([len(X_train.columns), 2, 4, 1])
        
        nn.fit(X_train, y_train)
        
        costs = nn.train(epochs = 1, learning_rate = 0.01)
        
        overall_costs[f'Iteration {i}'] = {f'Cost' : costs, 'Weights' : nn.weights, 'Biases' : nn.biases} 
        
        
    # Initialize variables to store information about the minimum cost
    min_cost = float('inf')
    min_cost_iteration = None

    # Iterate through the iterations
    for iteration_key, iteration_data in overall_costs.items():
        iteration_cost = iteration_data['Cost'][0]

        # Check if the current iteration has a lower cost than the minimum
        if iteration_cost < min_cost:
            min_cost = iteration_cost
            min_cost_iteration = iteration_key

    # Print the minimum cost and its corresponding iteration
    print(f'Minimum Cost: {min_cost} for {min_cost_iteration}')
    print(overall_costs[min_cost_iteration])