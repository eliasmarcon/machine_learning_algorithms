import pandas as pd
import numpy as np

## Unfinished code

class Node():
    
    def __init__(self, df : pd.DataFrame, node_type : str, depth : int, max_depth : int, min_samples : int): 
        
        self.__df = df
        self.__node_type = node_type
        self.__depth = depth
        self.__max_depth = max_depth
        self.__min_samples = min_samples
        
        self.__best_feature = None
        
        
    def _create_node(self) -> None:
        
        if self.__depth < self.__max_depth and self.__df.shape[0] >= self.__min_samples:
            
            # calculate the best feature and threshold to split the data
            best_feature, best_threshold = self._find_best_split()
            
            
            
    
    def _find_best_split(self):
        
        best_feature_name = None
        
        column_names = [col for col in self.__df.columns if col not in ['target']]
        
        for feature_name in column_names:
            
            # sort the dataframe by the feature
            self.__df = self.__df.sort_values(feature_name)
            
            # check if the feature is categorical
            if self.__df[feature_name].dtype == 'O':
        
                sum_squared_residuals = self.__calculate_ssr_categorical(feature_name)

            elif self.__df[feature_name].dtype in ['int64', 'float64']:
                
                sum_squared_residuals, best_split_value = self.__calculate_ssr_continuous(feature_name)
    
    
    def __calculate_ssr_categorical(self, feature_name : str) -> float:
        
        unique_values = self.__df[feature_name].unique()
        
        best_sum_squared_residuals = 0
        best_value = None
        
        for value in unique_values:
            
            sum_squared_residuals = 0
            
            left_values = self.__df[self.__df[feature_name] == value]
            right_values = self.__df[self.__df[feature_name] != value]
            
            left_residuals = left_values['target'] - left_values['target'].mean()
            right_residuals = right_values['target'] - right_values['target'].mean()
            
            sum_squared_residuals = (left_residuals**2).sum() + (right_residuals**2).sum()
        
            if sum_squared_residuals < best_sum_squared_residuals:
                
                best_sum_squared_residuals = sum_squared_residuals
                best_value = value
        
        return best_sum_squared_residuals, best_value
    
    def __calculate_ssr_continuous(self, feature_name : str) -> float:
        
        sum_squared_residuals = np.inf
        best_split_value = None
        
        for i in range(self.__df.shape[0]):
            
            left_node = self.__df.iloc[:i]
            right_node = self.__df.iloc[i:]
            
            left_residuals = left_node['target'] - left_node['target'].mean()
            right_residuals = right_node['target'] - right_node['target'].mean()
            
            current_ssr = (left_residuals**2).sum() + (right_residuals**2).sum()
            
            if current_ssr < sum_squared_residuals:
                
                sum_squared_residuals = current_ssr
                best_split_value = self.__df[feature_name].iloc[i]
                
        return sum_squared_residuals, best_split_value
    
        
class RegressionTree():
    
    def __init__(self, max_depth : int = None, min_samples : int = 2):
        
        # TODO : Add type checking for max_depth and min_samples
        
        self.__max_depth = max_depth
        self.__min_samples = min_samples
        self.__grown = False
        
    def fit(self, X : pd.DataFrame, y : pd.Series):
        
        # TODO : Add type checking for X and y
        
        if isinstance(X, pd.core.frame.DataFrame):
            self.__df = X.copy()
        else:
            self.__df = pd.DataFrame(X)

        self.__df['target'] = y.copy()
        
    def grow_tree(self) -> None:
        
        self.root_node = Node(self.__df, "root", 0, self.__max_depth, self.__min_samples)
        self.root_node._create_node()
        
        self.__grown = True
        
    def predict(self, X_test : pd.DataFrame) -> np.ndarray:
        
        if not isinstance(X_test, pd.core.frame.DataFrame):
            
            raise TypeError("X_test must be a pandas DataFrame")
        
        return self.root_node._predict(X_test)
    
    def print_tree(self) -> None:
        
        """
        Print the decision tree.

        Returns
        -------
        None
        """
        
        if self.__grown:
            self.root_node._print_tree()
        else:
            print("Tree has not been grown yet")
            
            