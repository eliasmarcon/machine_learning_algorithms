import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import sys

class Perceptron:
    
    def __init__(self, bias : int = 1) -> None:
        
        self.__bias = bias

    def fit(self, X_train : pd.DataFrame | np.ndarray | pd.core.series.Series, y_train : pd.DataFrame | np.ndarray | pd.core.series.Series) -> None:
        
        self.__X_train, self.__y_train, self.__number_w, self.__type_of_problem = self.__check_train_data(X_train, y_train)
        self.__weights = np.random.uniform(-1, 1, self.__number_w)
        
     
    def train(self, epochs : int = 100, learning_rate : float = 0.01, activation_function : str = None, loss_function : str = None, show_loss : bool = False, show_plots : bool = False) -> list:
        
        self.__check_lr_epochs_activation(epochs, learning_rate, activation_function, loss_function)
        
        self.__learning_rate = learning_rate
        self.__epochs = epochs
        
        if activation_function is None and self.__type_of_problem == "classification":
            self.__activation_function = "sigmoid"
        elif activation_function is None and self.__type_of_problem == "regression":
            self.__activation_function = "tan_h"
        else:
            self.__activation_function = activation_function
        
        if loss_function is None and self.__type_of_problem == "classification":
            self.__loss_function_type = "binary_cross_entropy"
        elif loss_function is None and self.__type_of_problem == "regression":
            self.__loss_function_type = "mse"
        else:
            self.__loss_function_type = loss_function
        
        if show_plots:
            self.__plot_data(trained = False)
               
        losses = []
        
        for epoch in range(1, self.__epochs + 1):
            
            # create mini batches
            mini_batches = self.__create_mini_batches()
            mini_batch_losses = []
            
            # iterate over mini batches
            for mini_batch in mini_batches:
                
                self.__X_train_mini_batch = mini_batch[:, :-1]
                self.__y_train_mini_batch = mini_batch[:, -1]
                
                self.__y_train_mini_batch = self.__y_train_mini_batch.reshape(-1, 1)
                                
                # Forward propagation
                y_train_pred_mini_batch = self.__forward_propagation()
                
                # Calculate the loss
                mini_batch_loss = self.__loss_function(y_train_pred_mini_batch, self.__y_train_mini_batch)
                mini_batch_losses.append(mini_batch_loss)
                
                # Backward propagation
                self.__backward_propagation(y_train_pred_mini_batch)
            
            epoch_loss = np.mean(mini_batch_losses)
            losses.append(epoch_loss)
            
            if show_loss and (epoch % 10 == 0 or epoch == 1):
                print("Loss after epoch %i: %f" %(epoch, epoch_loss))
        
        if show_plots:
            self.__plot_loss_history(losses)
            self.__plot_data(trained = True)
            
            
    def predict(self, X_test : pd.DataFrame | np.ndarray | pd.core.series.Series) -> np.ndarray:
        
        self.__check_test_data(X_test)
        
        # Forward propagation
        return self.__forward_propagation()
    
    
    def score(self, predictions : pd.DataFrame | np.ndarray | pd.core.series.Series, y_test : pd.DataFrame | np.ndarray | pd.core.series.Series) -> float:
        
        return self.__loss_function(predictions, y_test)
    
    
    def accuracy(self, predictions : pd.DataFrame | np.ndarray | pd.core.series.Series, y_test : pd.DataFrame | np.ndarray | pd.core.series.Series) -> float:
        
        if self.__type_of_problem == "classification":
            return np.mean(predictions == y_test)
        else:
            raise ValueError("Accuracy is only available for classification problems")

    #########################################################################################################################
    ############################################ Create Mini Batches ########################################################
    #########################################################################################################################
    def __create_mini_batches(self) -> list:
        
        # shuffle the data
        data = np.hstack((self.__X_train, self.__y_train.reshape(-1, 1)))
        np.random.shuffle(data)
        
        # create mini batches
        mini_batches = []
        batch_size = int(self.__X_train.shape[0] / 10)
        
        for i in range(0, self.__X_train.shape[0], batch_size):
            mini_batches.append(data[i:i + batch_size])
        
        return mini_batches
    
    #########################################################################################################################
    ############################################ Loss Function ##############################################################
    #########################################################################################################################
    def __loss_function(self, y_train_pred : np.ndarray, y_train_test : np.ndarray) -> float:
    
        if self.__type_of_problem == "classification":
            return self.__cross_entropy_loss(y_train_pred, y_train_test)
        
        elif self.__type_of_problem == "regression":
            
            if self.__loss_function_type == "mse":
                return self.__mse_loss(y_train_pred, y_train_test)
            
            elif self.__loss_function_type == "mae":
                return self.__mae_loss(y_train_pred, y_train_test)
    
    
    def __cross_entropy_loss(self, y_train_pred : np.ndarray, y_train_test : np.ndarray) -> np.ndarray:
        epsilon = 1e-7
        loss = - (y_train_test * np.log(y_train_pred + epsilon) + (1 - y_train_test) * np.log(1 - y_train_pred + epsilon))
        return np.mean(loss)
        
        
    def __mse_loss(self, y_train_pred : np.ndarray, y_train_test : np.ndarray) -> np.ndarray:
        losses = (y_train_pred - y_train_test)**2
        return np.mean(losses) # mean squared error
    
    
    def __mae_loss(self, y_train_pred : np.ndarray, y_train_test : np.ndarray) -> np.ndarray:
        losses = np.abs(y_train_pred - y_train_test)
        return np.mean(losses)
        
    #########################################################################################################################
    ############################################ Forward Propagation ########################################################
    #########################################################################################################################
    def __forward_propagation(self) -> float:
        
        x = self.__X_train_mini_batch @ self.__weights + self.__bias
        
        if self.__activation_function == "tan_h":
            return self.__hyperbolic_tangent_activation(x)
        
        elif self.__activation_function == "sigmoid":
            return self.__sigmoid_activation(x)
        
        
    def __sigmoid_activation(self, x : np.ndarray) -> np.ndarray:
            
        return 1 / (1 + np.exp(-x))
    
    
    def __hyperbolic_tangent_activation(self, x : np.ndarray) -> np.ndarray:
            
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    
    #########################################################################################################################
    ############################################ Backward Propagation #######################################################
    #########################################################################################################################
    def __backward_propagation(self, y_train_pred) -> None:
        
        # Compute the gradients
        dL_dw, dL_db = self.__compute_gradients(y_train_pred)
        
        # Update the weights and bias
        self.__weights -= self.__learning_rate * dL_dw
        self.__bias -= self.__learning_rate * dL_db
    
    
    def __sigmoid_activation_derivative(self, x : np.ndarray) -> np.ndarray:
            
        return np.exp(-x) / ((1 + np.exp(-x))**2) #self.__sigmoid_activation(x) * (1 - self.__sigmoid_activation(x))
    
    
    def __hyperbolic_tangent_activation_derivative(self, x : np.ndarray) -> np.ndarray:
        
        return 1 - self.__hyperbolic_tangent_activation(x)**2
    
    
    def __compute_gradients(self, y_train_pred : np.ndarray) -> tuple:
        
        dL_dw = np.dot(self.__X_train_mini_batch.T, (y_train_pred - self.__y_train_mini_batch))
        dL_db = np.sum(y_train_pred - self.__y_train_mini_batch)
        
        return dL_dw, dL_db
        
        
    #########################################################################################################################
    #########################################################################################################################
    #########################################################################################################################
    def __plot_data(self, trained : bool  = False) -> None:
        
        """
        This function plots the generated data.
        """
        
        if trained:
            title = 'Training Data with Trained Neuron (Sigmoid Activation)'
            label_titel = 'Decision Boundary (Trained)'
            
        else:
            title = 'Training Data with Untrained Neuron (Sigmoid Activation)' 
            label_titel = 'Decision Boundary (Untrained)'
            
        plt.figure(figsize=(12, 6))
        plt.scatter(self.__X_train, self.__y_train, label = 'Data')
        
        # Plot the decision boundary of the untrained neuron
        x_decision_boundary = np.linspace(min(self.__X_train), max(self.__X_train), len(self.__X_train))
        y_decision_boundary = self.sigmoid(self.__weights * x_decision_boundary + self.__bias)
        
        plt.plot(x_decision_boundary, y_decision_boundary, color='red', label=label_titel)
        
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.title(title)
        plt.tight_layout()
        plt.show()
             
                
    def __plot_loss_history(self, losses) -> None:
        
        plt.figure(figsize=(12, 6)) 

        plt.scatter(range(0, self.epochs), losses, s = 8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss History on training data')
        plt.tight_layout()
        plt.show()

    ############################################################################################################
    ########################################### Validating Data ################################################
    ############################################################################################################
    def __check_train_data(self, X_train : pd.DataFrame | np.ndarray | pd.core.series.Series, y_train : pd.DataFrame | np.ndarray | pd.core.series.Series) -> tuple:
        
        if not isinstance(X_train, (pd.DataFrame, np.ndarray, pd.core.series.Series)):
            raise TypeError("X_train must be a pandas DataFrame/series or numpy array")
        
        if not isinstance(y_train, (pd.DataFrame, np.ndarray, pd.core.series.Series)):
            raise TypeError("y_train must be a pandas DataFrame/series or numpy array")
        
        if isinstance(X_train, (pd.DataFrame, pd.core.series.Series)):
            X_train = X_train.to_numpy()

        if isinstance(y_train, (pd.DataFrame, pd.core.series.Series)):
            y_train = y_train.to_numpy()
            
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train must have the same length")
            
        # check if regression or classification
        if len(np.unique(y_train)) > 2:
            type_of_problem = "regression"
        else:
            type_of_problem = "classification"

        return (X_train, y_train, X_train.shape[1], type_of_problem)
    
    
    def __check_lr_epochs_activation(self, epochs : int, learning_rate : float, activation_function : str, loss_function : str) -> None:
        
        if not isinstance(epochs, int):
            raise TypeError("epochs must be an integer")
        
        if not isinstance(learning_rate, float):
            raise TypeError("learning_rate must be a float")
        
        if not isinstance(activation_function, str) and activation_function is not None:
            raise TypeError("activation_function must be a string")
        
        if epochs < 1:
            raise ValueError("epochs must be greater than 0")
        
        if activation_function not in ["tan_h", "sigmoid", None]:
            raise ValueError("activation_function must be either tan_h or sigmoid")

        if loss_function not in ["mse", "mae", "binary_cross_entropy", None]:
            raise ValueError("loss_function must be either mse, mae or binary_cross_entropy")
        
        if self.__type_of_problem == "classification" and loss_function not in ["binary_cross_entropy", None]:
            raise ValueError("loss_function must be binary_cross_entropy for classification")
        
        if self.__type_of_problem == "regression" and loss_function not in ["mse", "mae", None]:
            raise ValueError("loss_function must be mse or mae for regression")


    def __check_test_data(self, X_test : pd.DataFrame | np.ndarray | pd.core.series.Series) -> np.ndarray:
        
        if not isinstance(X_test, (pd.DataFrame, np.ndarray, pd.core.series.Series)):
            raise TypeError("X_test must be a pandas DataFrame/series or numpy array")
        
        if isinstance(X_test, (pd.DataFrame, pd.core.series.Series)):
            X_test = X_test.to_numpy()
        
        if X_test.shape[1] != self.__number_w:
            raise ValueError("X_test must have the same number of features as X_train")
        
        return X_test



############################################################################################################
############################################## Example Usage ###############################################
############################################################################################################
if __name__ == "__main__":
    
    if len(sys.argv) == 2:
        epochs = int(sys.argv[1])
    else:
        epochs = 1000
    
    for epoch in range(100, epochs, 100):
    
        for lr in [0.001, 0.01, 0.1, 1]:
            
            single_neuron = Perceptron()
            
            print("Epochs: %i, Learning Rate: %f" %(epoch, lr))
            single_neuron.train(epochs = epoch, learning_rate = lr, show_loss = False, show_plots = True)


    # best current Epochs 200, Learning Rate 1
    # single_neuron = Neuron()
    # single_neuron.train(epochs = 200, learning_rate = 1, show_loss = False, show_plots = True)