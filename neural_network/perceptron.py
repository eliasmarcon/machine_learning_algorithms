import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import sys

class Perceptron:
    
    def __init__(self) -> None:
        
        self.__bias = 0

    def fit(self, X_train : pd.DataFrame | np.ndarray | pd.core.series.Series, y_train : pd.DataFrame | np.ndarray | pd.core.series.Series) -> None:
        
        self.__X_train, self.__y_train, self.__number_w, self.__type_of_problem = self.__check_train_data(X_train, y_train)
        self.__weights = np.random.uniform(-1, 1, self.__number_w)
        
     
    def train(self, loss_function : str, epochs : int = 100, learning_rate : float = 0.01, activation_function : str = "relu", show_loss : bool = False, show_plots : bool = False) -> list:
        
        self.__check_lr_epochs_activation(epochs, learning_rate, activation_function, loss_function)
        
        self.__learning_rate = learning_rate
        self.__epochs = epochs
        self.__activation_function = activation_function
        
        
        if show_plots:
            self.__plot_data(trained = False)
               
        losses = []
        
        for epoch in range(1, self.__epochs + 1):
            
            # Forward propagation
            y_pred = self.__forward_propagation()
            
            
            epoch_loss = self.__backward_propagation(y_pred)
            losses.append(epoch_loss)
            
            if show_loss and (epoch % 10 == 0 or epoch == 1):
                print("Loss after epoch %i: %f" %(epoch, epoch_loss))
        
        if show_plots:
            self.__plot_loss_history(losses)
            self.__plot_data(trained = True)
            
            
    def predict(self, X_test : pd.DataFrame | np.ndarray | pd.core.series.Series) -> np.ndarray:
        
        self.__check_test_data(X_test)
        
        
    
    
    #########################################################################################################################
    #########################################################################################################################
    #########################################################################################################################
    def __forward_propagation(self) -> float:
        
        if self.__activation_function == "relu":
            return np.maximum(0, self.__weights * self.__X_train + self.__bias)
        
        elif self.__activation_function == "sigmoid":
            return 1 / (1 + np.exp(-self.__weights * self.__X_train - self.__bias))
    
    def __backward_propagation(self, y_pred : float) -> float:
        
        # Calcualte the loss
        epoch_loss = np.mean(self.__loss_function(y_pred, self.__y_train))
        
        # Compute the gradients
        dL_dw, dL_db = self.__compute_gradients(y_pred)
        
        # Update the weights and bias
        self.__weights -= self.learning_rate * np.mean(dL_dw)
        self.__bias -= self.learning_rate * np.mean(dL_db)      
        
        return epoch_loss  
            
    def __compute_gradients(self, y_pred : float) -> tuple:
        
        dL_dw = -2 * (self.__y_train - y_pred) * (self.__X_train * np.exp(-self.__weights * self.__X_train - self.__bias)) / (1 + np.exp(-self.__weights * self.__X_train - self.__bias))**2
        dL_db = -2 * (self.__y_train - y_pred) * (np.exp(-self.__weights * self.__X_train - self.__bias)) / (1 + np.exp(-self.__weights * self.__X_train - self.__bias))**2
        
        return (dL_dw, dL_db)
    
    def __loss_function(self, y : int, y_true : int) -> int:
    
        if self.__type_of_problem == "classification":
            return -y_true * np.log(y) - (1 - y_true) * np.log(1 - y) # binary cross entropy
        
        elif self.__type_of_problem == "regression":
            if self.__loss_function == "mse":
                return (y - y_true)**2
            elif self.__loss_function == "mae":
                return np.abs(y - y_true)
            
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
            
        if len(X_train.shape[0]) != len(y_train.shape[0]):
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
        
        if not isinstance(activation_function, str):
            raise TypeError("activation_function must be a string")
        
        if epochs < 1:
            raise ValueError("epochs must be greater than 0")
        
        if activation_function not in ["relu", "sigmoid"]:
            raise ValueError("activation_function must be either relu or sigmoid")

        if loss_function not in ["mse", "mae", "binary_cross_entropy"]:
            raise ValueError("loss_function must be either mse, mae or binary_cross_entropy")
        
        if self.__type_of_problem == "classification" and loss_function not in ["binary_cross_entropy"]:
            raise ValueError("loss_function must be binary_cross_entropy for classification")
        
        if self.__type_of_problem == "regression" and loss_function not in ["mse", "mae"]:
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
############################################### Main Function ä#############################################
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


