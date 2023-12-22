import numpy as np
import random
import matplotlib.pyplot as plt
import sys

class Neuron:
    
    def __init__(self) -> None:
        
        self.w = np.random.uniform(-1, 1)
        self.b = 0
        
        self.generate_data()
     
    def train(self, epochs : int = 100, learning_rate : float = 0.01, show_loss : bool = False, show_plots : bool = False) -> list:
        
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        if show_plots:
            self.plot_data(trained = False)
        
        losses = []
        
        for epoch in range(1, self.epochs + 1):
            
            # Forward propagation
            y_pred = self.forward_propagation()
            
            # Calcualte the loss
            epoch_loss = np.mean(self.loss_function(y_pred, self.y))
            losses.append(epoch_loss)
            
            # Compute the gradients
            dL_dw = -2 * (self.y - y_pred) * (self.X * np.exp(-self.w * self.X - self.b)) / (1 + np.exp(-self.w * self.X - self.b))**2
            dL_db = -2 * (self.y - y_pred) * (np.exp(-self.w * self.X - self.b)) / (1 + np.exp(-self.w * self.X - self.b))**2
            
            # Update the weights and bias
            self.w -= self.learning_rate * np.mean(dL_dw)
            self.b -= self.learning_rate * np.mean(dL_db)
        
            if show_loss and (epoch % 10 == 0 or epoch == 1):
                print("Loss after epoch %i: %f" %(epoch, epoch_loss))
        
        if show_plots:
            self.plot_loss_history(losses)
            self.plot_data(trained = True)
            
    def forward_propagation(self) -> float:

        return self.sigmoid(self.w * self.X + self.b)
     
    def generate_data(self) -> None:
        
        """
        This function generates artificial data for the exercise.
        """

        self.X = np.empty(100)
        self.y = np.empty(100)
        
        for i in range(100):
            if random.random() < 0.5:
                self.X[i] = np.random.normal(loc=-1.25, scale=0.75)
                self.y[i] = 0
            else:
                self.X[i] = np.random.normal(loc=1.25, scale=0.75)
                self.y[i] = 1
    
    def loss_function(self, y : int, y_true : int) -> int:
    
        return (y - y_true)**2
    
    def sigmoid(self, x : float) -> float:
        
        """
        This function calculates the logistic function.
        """
        return 1 / (1 + np.exp(-x))
         
    #########################################################################################################################
    #########################################################################################################################
    #########################################################################################################################
    
    def plot_data(self, trained : bool  = False) -> None:
        
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
        plt.scatter(self.X, self.y, label = 'Data')
        
        # Plot the decision boundary of the untrained neuron
        x_decision_boundary = np.linspace(min(self.X), max(self.X), len(self.X))
        y_decision_boundary = self.sigmoid(self.w * x_decision_boundary + self.b)
        
        plt.plot(x_decision_boundary, y_decision_boundary, color='red', label=label_titel)
        
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.title(title)
        plt.tight_layout()
        plt.show()
             
                
    def plot_loss_history(self, losses) -> None:
        
        plt.figure(figsize=(12, 6)) 

        plt.scatter(range(0, self.epochs), losses, s = 8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss History on training data')
        plt.tight_layout()
        plt.show()






if __name__ == "__main__":
    
    if len(sys.argv) == 2:
        epochs = int(sys.argv[1])
    else:
        epochs = 1000
    
    for epoch in range(100, epochs, 100):
    
        for lr in [0.001, 0.01, 0.1, 1]:
            
            single_neuron = Neuron()
            
            print("Epochs: %i, Learning Rate: %f" %(epoch, lr))
            single_neuron.train(epochs = epoch, learning_rate = lr, show_loss = False, show_plots = True)


    # best current Epochs 200, Learning Rate 1
    # single_neuron = Neuron()
    # single_neuron.train(epochs = 200, learning_rate = 1, show_loss = False, show_plots = True)


