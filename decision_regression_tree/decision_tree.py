import pandas as pd
import numpy as np
from collections import Counter

############################################################################################################
########################################### Node Class #####################################################
############################################################################################################
class Node():
    
    def __init__(self, df : pd.DataFrame, node_type : str, depth : int, max_depth : int, min_samples : int):
        
        """
        A class representing a node in a decision tree.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing the data for this node.
        node_type : str
            The type of the node ('root', 'left', 'right').
        depth : int
            The depth of the node in the tree.
        max_depth : int
            The maximum depth allowed for the tree.
        min_samples : int
            The minimum number of samples required to split a node.

        Returns
        -------
        None
        """
        
        self.__df = df
        self.__node_type = node_type
        self.__node_type_2 = None
        self.__depth = depth
        self.__max_depth = max_depth
        self.__min_samples = min_samples
        self.__best_feature = None
        self.__best_gini = None
        self.__class_distribution = None
        self.__most_common = None
        self.__left = None
        self.__right = None
        
    def _create_node(self) -> None:
        
        """
        Create the decision tree recursively.

        Returns
        -------
        None
        """
        
        if self.__depth < self.__max_depth and len(self.__df) >= self.__min_samples:
        
            best_feature_name, best_gini, best_feature_value_continuous = self.__best_split()
            
            # Check if splitting is wise
            gini_without_split = self.__calculate_gini_impurity_without_split()
            
            if gini_without_split < best_gini:
                self.__node_type_2 = 'leaf'
                self.__class_distribution = Counter(self.__df['target'])
                self.__most_common = self.__class_distribution.most_common(1)[0][0]
                self.__best_feature = "No Splitting necessary - Gini impurity is lower without split"
            
            else:
                
                if best_feature_value_continuous is not None:
                    left_df, right_df = self.__split_continuous(best_feature_name, best_feature_value_continuous)
                else:
                    left_df, right_df = self.__split_categorical(best_feature_name)
                    
                self.__best_feature = best_feature_name
                self.__best_gini = best_gini
                self.__class_distribution = Counter(self.__df['target'])
                self.__most_common = self.__class_distribution.most_common(1)[0][0]
                
                left_node = Node(left_df, 'left', self.__depth + 1, self.__max_depth, self.__min_samples)
                right_node = Node(right_df, 'right', self.__depth + 1, self.__max_depth, self.__min_samples)
                
                if len(left_df) > 0:
                    self.__left = left_node
                    self.__left._create_node()
                    
                if len(right_df) > 0:
                    self.__right = right_node
                    self.__right._create_node()
        
        else:
            
            if self.__depth == self.__max_depth:
                self.__best_feature = "Max Depth reached"
            elif len(self.__df) < self.__min_samples:
                self.__best_feature = "Not enough samples"
                
            self.__best_gini = None
            self.__class_distribution = Counter(self.__df['target'])
            self.__most_common = self.__class_distribution.most_common(1)[0][0]
                
                
    ############################################################################################################
    ########################################### Split Function #################################################
    ############################################################################################################
    def __best_split(self) -> tuple:
        
        """
        Find the best feature to split on.

        Returns
        -------
        tuple
            A tuple containing the best feature name, best Gini impurity, and best feature value for continuous features.
        """
        
        best_gini_value = 1.0  # Initialize the best Gini impurity with a high value
        best_feature_name = None  # Initialize the best feature name as None
        best_feature_value_continuous = None  # Initialize the best feature value for continuous features
        
        column_names = [col for col in self.__df.columns if col not in ['target']]
        
        # Iterate through all the features (columns) in the DataFrame
        for feature_name in column_names:
        
            # check if the feature is categorical
            if self.__df[feature_name].dtype == 'O':
        
                gini_impurity_feature = self.__calculate_gini_categorical(feature_name)

            elif self.__df[feature_name].dtype in ['int64', 'float64']:
                
                gini_impurity_feature, best_split_value = self.__calculate_gini_continuous(feature_name)

            # Check if this is the best split so far (lowest Gini impurity)
            if gini_impurity_feature < best_gini_value and self.__df[feature_name].dtype == 'O':
        
                best_gini_value = gini_impurity_feature
                best_feature_name = feature_name
                
            elif gini_impurity_feature < best_gini_value and self.__df[feature_name].dtype in ['int64', 'float64']:
                
                best_gini_value = gini_impurity_feature
                best_feature_name = feature_name
                best_feature_value_continuous = best_split_value
        
        # Return the best feature for splitting and its associated Gini impurity
        if best_feature_value_continuous != None:
            return (best_feature_name, best_gini_value, best_feature_value_continuous)
        else:
            return (best_feature_name, best_gini_value, None)
        
    def __split_categorical(self, feature_name) -> tuple:
        
        """
        Split the DataFrame based on a categorical feature.

        Parameters
        ----------
        feature_name : str
            The name of the categorical feature.

        Returns
        -------
        tuple
            A tuple containing the left and right subsets of the DataFrame after the split.
        """
        
        return (self.__df[self.__df[feature_name] == "Yes"].copy().drop(feature_name, axis=1), self.__df[self.__df[feature_name] == "No"].copy().drop(feature_name, axis=1))
    
    def __split_continuous(self, feature_name, best_feature_value_continuous) -> tuple:
        
        """
        Split the DataFrame based on a continuous feature.

        Parameters
        ----------
        feature_name : str
            The name of the continuous feature.
        best_feature_value_continuous : float
            The best value for splitting the continuous feature.

        Returns
        -------
        tuple
            A tuple containing the left and right subsets of the DataFrame after the split.
        """
        
        return (self.__df[self.__df[feature_name] <= best_feature_value_continuous].copy().drop(feature_name, axis=1), self.__df[self.__df[feature_name] > best_feature_value_continuous].copy().drop(feature_name, axis=1))
        

    ############################################################################################################
    ########################################### Gini Function ##################################################
    ############################################################################################################    
    def __calculate_gini_categorical(self, feature_name) -> str:
        
        """
        Calculate Gini impurity for a categorical feature.

        Parameters
        ----------
        feature_name : str
            The name of the categorical feature.

        Returns
        -------
        float
            The Gini impurity for the split.
        """
        
        # Count the number of 'Yes' and 'No' labels in the left and right branches
        left_counter = Counter(self.__df[self.__df[feature_name] == "Yes"]['target'])
        right_counter = Counter(self.__df[self.__df[feature_name] == "No"]['target'])

        # Extract the counts of 'Yes' and 'No' for left and right branches
        count_left_yes, count_left_no = self.__get_counts(left_counter)
        count_right_yes, count_right_no = self.__get_counts(right_counter)
        
        left_gini_impurity = self.__calculate_gini_impurity(count_left_yes, count_left_no)
        right_gini_impurity = self.__calculate_gini_impurity(count_right_yes, count_right_no)
        
        # Calculate the Gini impurity for the split
        gini_impurity_feature = self.__calculate_gini_impurity_split(left_gini_impurity, right_gini_impurity, count_left_yes, count_left_no, count_right_yes, count_right_no)
        
        return gini_impurity_feature
        
    def __calculate_gini_continuous(self, feature_name) -> tuple:
        
        """
        Calculate Gini impurity for a continuous feature.

        Parameters
        ----------
        feature_name : str
            The name of the continuous feature.

        Returns
        -------
        tuple
            A tuple containing the best Gini impurity and the best split value for the continuous feature.
        """
        
        # Get the unique values in the feature
        unique_values = self.__df[feature_name].unique()
        
        # Sort the unique values
        unique_values.sort()
        
        best_gini_temp = 1.0
        best_split_value = None

        # Iterate through all the unique values
        for i in range(1, len(unique_values)):
            
            # Calculate the potential split value
            split_value = (unique_values[i - 1] + unique_values[i]) / 2
            
            # Split the DataFrame into two subsets
            left_subset = self.__df[self.__df[feature_name] <= split_value]
            right_subset = self.__df[self.__df[feature_name] > split_value]
            
            # Calculate the Gini impurity for the split
            gini = self.calculate_gini_continuous_split(left_subset, right_subset)
            
            # Check if this is the best split so far
            if gini < best_gini:
                best_gini = gini
                best_split_value = split_value
        
        return (best_gini, best_split_value)
    
    def __calculate_gini_impurity(self, count_yes, count_no) -> float:
        
        """
        Calculate Gini impurity.

        Parameters
        ----------
        count_yes : int
            The count of 'Yes' instances.
        count_no : int
            The count of 'No' instances.

        Returns
        -------
        float
            The Gini impurity.
        """
        
        # Calculate the total number of instances
        total = count_yes + count_no

        if total == 0:
            return 0
        else:
            # Calculate the Gini impurity
            return 1 - (count_yes / total)**2 - (count_no / total)**2 # 1 - (prob_yes**2 + prob_no**2)
        
    def __calculate_gini_impurity_split(self, left_gini_impurity, right_gini_impurity, count_left_yes, count_left_no, count_right_yes, count_right_no) -> float:
        
        """
        Calculate Gini impurity for a split.

        Parameters
        ----------
        left_gini_impurity : float
            Gini impurity of the left split.
        right_gini_impurity : float
            Gini impurity of the right split.
        count_left_yes : int
            Count of 'Yes' instances in the left split.
        count_left_no : int
            Count of 'No' instances in the left split.
        count_right_yes : int
            Count of 'Yes' instances in the right split.
        count_right_no : int
            Count of 'No' instances in the right split.

        Returns
        -------
        float
            The Gini impurity for the split.
        """
        
        # Calculate the total number of instances
        total = count_left_yes + count_left_no + count_right_yes + count_right_no
        
        # Calculate the Gini impurity for the split
        return ( (count_left_yes + count_left_no) / total * left_gini_impurity ) + ( (count_right_yes + count_right_no) / total * right_gini_impurity )
    
    def __calculate_gini_impurity_without_split(self) -> float:
        
        """
        Calculate Gini impurity without splitting.

        Returns
        -------
        float
            The Gini impurity.
        """
        
        target_counts = Counter(self.__df['target'])
        count_left_yes, count_left_no = self.__get_counts(target_counts)
        
        return self.__calculate_gini_impurity(count_left_yes, count_left_no)
    
    def __get_counts(self, counts) -> tuple:

        """
        Get counts of 'Yes' and 'No'.

        Parameters
        ----------
        counts : collections.Counter
            Counter object containing counts of 'Yes' and 'No'.

        Returns
        -------
        tuple
            A tuple containing the counts of 'Yes' and 'No'.
        """

        return (counts['Yes'], counts['No'])
    
    
    ############################################################################################################
    ########################################### Prediction Function ############################################
    ############################################################################################################    
    def _predict(self, X_test : pd.core.frame.DataFrame) -> list:
        
        """
        Predict the class label for the given test data.

        Parameters
        ----------
        X_test : pandas.DataFrame
            The test data for prediction.

        Returns
        -------
        list
            A list containing the predicted class labels.
        """
        
        predictions = []

        for _, x in X_test.iterrows():

            values = {}

            for feature in X_test.columns:

                values.update({feature: x[feature]})

            cur_node = self  # Reset cur_node for each observation

            while cur_node.__depth < cur_node.__max_depth:

                # Traversing the nodes all the way to the bottom
                best_feature = cur_node.__best_feature

                if values.get(best_feature) == "Yes":

                    if cur_node.__left is not None:

                        cur_node = cur_node.__left

                elif values.get(best_feature) == "No":

                    if cur_node.__right is not None:
                        
                        cur_node = cur_node.__right
                else:
                    break

            predictions.append(cur_node.__most_common)

        return predictions
    
    
    ############################################################################################################
    ########################################### Print Function #################################################
    ############################################################################################################
    def __print_info(self, width : int = 4) -> None:
        
        """
        Print information about the node.

        Parameters
        ----------
        width : int, optional
            Width parameter for printing, by default 4.

        Returns
        -------
        None
        """

        const = int(self.__depth * width ** 1.5)
        spaces = "-" * const

        if self.__node_type == 'root':
            print(f"Node Type: {self.__node_type}")

        else:
            print(f"|{spaces} Node Type: {self.__node_type} at depth {self.__depth}")
        
        print(f"{' ' * const}   | Best Feature: {self.__best_feature}")
        
        if self.__best_gini is None:
            print(f"{' ' * const}   | GINI impurity of the node: None")

        else:
            print(f"{' ' * const}   | GINI impurity of the node: {round(self.__best_gini, 2)}")

        print(f"{' ' * const}   | Class distribution in the node: {self.__class_distribution}")
        print(f"{' ' * const}   | Predicted class: {self.__most_common}")
        print(f"{' ' * const}   | Number of samples in the node: {len(self.__df)}")
        
        if self.__depth == self.__max_depth or self.__node_type_2 == 'leaf':
            print(f"{' ' * const}   | Leaf node: Yes\n")
        else:
            print(f"{' ' * const}   | Leaf node: No\n")

    def _print_tree(self) -> None:
        
        """
        Print the decision tree.

        Returns
        -------
        None
        """
        
        self.__print_info() 
        
        if self.__left is not None: 
            self.__left._print_tree()
        
        if self.__right is not None:
            self.__right._print_tree()



############################################################################################################
########################################### Tree Class #####################################################
############################################################################################################
class DecisionTree():
    
    """
    A class representing a decision tree.

    Parameters
    ----------
    max_depth : int, optional
        The maximum depth of the tree, by default None.
    num_min_samples : int, optional
        The minimum number of samples required to split a node, by default 20.

    Returns
    -------
    None
    """
    
    def __init__(self, max_depth : int = None, num_min_samples : int = 20):
        
        # function to check the parameters
        self.__check_init_params(max_depth, num_min_samples)
        
        self.__max_depth = max_depth  
        self.__num_min_samples = num_min_samples  
        self.__grown = False
        
    def fit(self, X : pd.core.frame.DataFrame | pd.core.series.Series | np.ndarray, y : pd.core.frame.DataFrame | pd.core.series.Series | np.ndarray) -> None:
        
        """
        Fit the decision tree to the given data.

        Parameters
        ----------
        X : pandas.DataFrame, pandas.Series, numpy.ndarray
            The input features.
        y : pandas.DataFrame, pandas.Series, numpy.ndarray
            The target variable.

        Returns
        -------
        None
        """
        
        # function to check the parameters
        self.__check_fit_params(X, y)
        
        if isinstance(X, pd.core.frame.DataFrame):
            self.__df = X.copy()
        else:
            self.__df = pd.DataFrame(X)

        self.__df['target'] = y.copy()
        
    def grow_tree(self) -> None:
        
        """
        Grow the decision tree.

        Returns
        -------
        None
        """
        
        self.root_node = Node(self.__df, 'root', 0, max_depth = self.__max_depth, min_samples = self.__num_min_samples)
        self.root_node._create_node()
        self.__grown = True

    def predict(self, X_test : pd.DataFrame) -> np.ndarray:
        
        """
        Predict the class labels for the given test data.

        Parameters
        ----------
        X_test : pandas.DataFrame
            The test data.

        Returns
        -------
        numpy.ndarray
            An array containing the predicted class labels.
        """
        
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
    
    
    ############################################################################################################
    ########################################### Check Function #################################################
    ############################################################################################################
    def __check_init_params(self, max_depth : int, num_min_samples : int) -> None:
        
        """
        Check the validity of initialization parameters.

        Parameters
        ----------
        max_depth : int
            The maximum depth of the tree.
        num_min_samples : int
            The minimum number of samples required to split a node.

        Returns
        -------
        None
        """
        
        # Check if max_depth is a valid type
        if not isinstance(max_depth, int) and max_depth is not None:
            raise TypeError("max_depth must be an integer")
        
        # Check if max_depth is a valid value
        if max_depth is not None and max_depth <= 0:
            raise ValueError("max_depth must be greater than 0")
        
        # Check if num_min_samples is a valid type
        if not isinstance(num_min_samples, int):
            raise TypeError("num_min_samples must be an integer")
        
        # Check if num_min_samples is a valid value
        if num_min_samples <= 0:
            raise ValueError("num_min_samples must be greater than 0")
     
    def __check_fit_params(self, X : pd.core.frame.DataFrame | pd.core.series.Series | np.ndarray, y : pd.core.frame.DataFrame | pd.core.series.Series | np.ndarray) -> None:
        
        """
        Check the validity of fit parameters.

        Parameters
        ----------
        X : numpy.ndarray
            The input features.
        y : numpy.ndarray
            The target variable.

        Returns
        -------
        None
        """
        
        # Check if data is a valid type
        if not isinstance(X, (pd.core.frame.DataFrame, pd.core.series.Series, np.ndarray)):
            raise TypeError("X must be a pandas DataFrame/Series or a numpy array")
        
        if not isinstance(y, (pd.core.frame.DataFrame, pd.core.series.Series, np.ndarray)):
            raise TypeError("y must be a pandas DataFrame/Series or a numpy array")
        
        # Check if the data has the same number of rows
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")

    

############################################################################################################
############################################## Example Usage ###############################################
############################################################################################################
if __name__ == "__main__":
    
    import random
    from sklearn.model_selection import train_test_split
    
    # Set the random seed for reproducibility
    random.seed(42)

    # Define the number of rows
    num_rows = 15000

    # Create a dictionary to store the data for each column
    data = {
        'chest_pain': [random.choice(['Yes', 'Yes', 'No']) for _ in range(num_rows)],
        'good_blood_circulation': [random.choice(['Yes', 'No', 'No']) for _ in range(num_rows)],
        'blocked_arteries': [random.choice(['Yes', 'No', 'No']) for _ in range(num_rows)],
        'overweight': [random.choice(['Yes', 'Yes', 'No']) for _ in range(num_rows)],
        'high_blood_pressure': [random.choice(['Yes', 'Yes', 'Yes', 'No']) for _ in range(num_rows)],
        'heart_disease': [random.choice(['Yes', 'No']) for _ in range(num_rows)]
    }

    # Create a pandas DataFrame from the dictionary
    df = pd.DataFrame(data)
    
    X_train, X_test, y_train, y_test = train_test_split(df.drop('heart_disease', axis=1), df['heart_disease'], test_size=0.2, random_state=42)
    
    decision_tree = DecisionTree(max_depth = 4, num_min_samples = 20)
    decision_tree.fit(X_train, y_train)

    # Grow the decision tree
    decision_tree.grow_tree()
    
    # Print the decision tree
    decision_tree.print_tree()
    
    predictions = decision_tree.predict(df.drop('heart_disease', axis=1))

    # compare the predictions to the actual values
    accuracy = (predictions == df['heart_disease']).mean()
    print(f"The accuracy of the model is {accuracy * 100:.2f}%")

