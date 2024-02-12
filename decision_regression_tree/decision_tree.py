import pandas as pd
import numpy as np
import random
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
class Node():

    def __init__(self, max_depth=None, depth=None, num_min_samples=20, node_type=None, left=None, right=None):
        
        # Constructor for the Node class
        # Initialize the attributes with default or provided values
        self.max_depth = max_depth  # Maximum depth for the tree
        self.depth = depth if depth else 0  # Current depth of this node, default to 0
        self.num_min_samples = num_min_samples  # Minimum number of samples required for splitting
        self.node_type = node_type if node_type else 'root'  # Type of node (root, left, or right), default to 'root'
        self.left = left  # Reference to the left child node
        self.right = right  # Reference to the right child node

    def fit(self, X, y):
        
        # Method to fit (store) the training data (X) and labels (y) in the node's dataframe (df)
        df = X.copy()
        df['target'] = y  # Create a new column in the DataFrame for labels
        
        self.df = df  # Store the DataFrame in the node

    def grow_tree(self):
        
        # Method to grow the decision tree node by node
        if self.depth < self.max_depth and len(self.df) >= self.num_min_samples:
        
            best_feature_name, best_gini = self.best_split()  # Find the best feature to split and its Gini impurity
            gini_without_spit = self.gini_impurity_before_split()  # Calculate Gini impurity without splitting

            if best_gini < gini_without_spit and best_feature_name is not None:
        
                # If splitting is beneficial, create left and right child nodes
                self.best_feature = best_feature_name
                self.best_gini = best_gini
                self.class_distribution = Counter(self.df['target'])  # Count class distribution
                self.most_common = self.class_distribution.most_common(1)[0][0]  # Get the most common class

                if len(self.df.columns) == 2:
        
                    # Handle a DataFrame with two columns (features), "Yes" and "No"
                    left_df = self.df[self.df[best_feature_name] == "Yes"].copy()
                    right_df = self.df[self.df[best_feature_name] == "No"].copy()
        
                else:
        
                    # Handle a DataFrame with more than two columns
                    left_df = self.df[self.df[best_feature_name] == "Yes"].copy().drop(best_feature_name, axis=1)
                    right_df = self.df[self.df[best_feature_name] == "No"].copy().drop(best_feature_name, axis=1)

                # Create left and right child nodes
                left_node = Node(max_depth=self.max_depth, depth=self.depth + 1, node_type='left')
                right_node = Node(max_depth=self.max_depth, depth=self.depth + 1, node_type='right')

                if len(left_df) != 0:
        
                    left_node.fit(left_df.iloc[:, :-1], left_df.iloc[:, -1])
                    self.left = left_node
                    self.left.grow_tree()

                if len(right_df) != 0:
        
                    right_node.fit(right_df.iloc[:, :-1], right_df.iloc[:, -1])
                    self.right = right_node
                    self.right.grow_tree()
            else:
        
                # If splitting is not beneficial, make this node a leaf node
                self.best_feature = "No Splitting necessary Node is not purer than its parent node"
                # self.best_gini = None
                # self.class_distribution = Counter(self.df['target'])
                # self.most_common = self.class_distribution.most_common(1)[0][0]
        else:
        
            # If the maximum depth is reached or there are not enough samples, make this node a leaf node
            self.best_feature = "Max Depth reached or not enough samples"
            self.best_gini = None
            self.class_distribution = Counter(self.df['target'])
            self.most_common = self.class_distribution.most_common(1)[0][0]


    def best_split(self):

        # Method to find the best feature for splitting and its associated Gini impurity
        best_gini = 1.1  # Initialize the best Gini impurity with a high value
        best_feature_name = None  # Initialize the best feature name as None

        column_names = [col for col in self.df.columns if col not in ['target']]
        
        # Iterate through all the features (columns) in the DataFrame
        for feature_name in column_names:
        
            # Count the number of 'Yes' and 'No' labels in the left and right branches
            left_counter = Counter(self.df[self.df[feature_name] == "Yes"]['target'])
            right_counter = Counter(self.df[self.df[feature_name] == "No"]['target'])

            # Extract the counts of 'Yes' and 'No' for left and right branches
            count_left_yes, count_left_no = self.get_counts(left_counter)
            count_right_yes, count_right_no = self.get_counts(right_counter)

            # Calculate Gini impurity for left and right branches
            gini_left = self.gini_impurity(count_left_yes, count_left_no)
            gini_right = self.gini_impurity(count_right_yes, count_right_no)

            # Calculate the weighted Gini impurity for the potential split
            n = count_left_yes + count_left_no + count_right_yes + count_right_no
            overall_gini = ((count_left_yes + count_left_no) / n * gini_left) + ((count_right_yes + count_right_no) / n * gini_right)

            # Check if this is the best split so far (lowest Gini impurity)
            if overall_gini < best_gini:
        
                best_gini = overall_gini
                best_feature_name = feature_name
        
        # Return the best feature for splitting and its associated Gini impurity
        return best_feature_name, best_gini

    def gini_impurity(self, count_yes, count_no):

        # Method to calculate Gini impurity given counts of 'Yes' and 'No' labels
        n = count_yes + count_no  # Total number of observations
        
        # If there are no observations, return the lowest possible Gini impurity (0.0)
        if n == 0:
            return 0.0
        else:
            # Calculate and return the Gini impurity
            return 1 - (count_yes / n)**2 - (count_no / n)**2

    def gini_impurity_before_split(self):
        
        # Method to calculate Gini impurity for the current node before any split
        target_counter = Counter(self.df['target'])  # Count the class distribution
        count_left_yes, count_left_no = self.get_counts(target_counter)  # Get counts of 'Yes' and 'No' labels
        gini_without_spit = self.gini_impurity(count_left_yes, count_left_no)  # Calculate Gini impurity
        
        return gini_without_spit  # Return Gini impurity before any split

    def get_counts(self, counts):

        # Method to extract counts of 'Yes' and 'No' from the Counter object
        yes, no = counts['Yes'], counts['No']
        
        return yes, no  # Return the counts of 'Yes' and 'No'


    def print_info(self, width=4):
        
        """
        Method to print information about the current node.
        """
        
        const = int(self.depth * width ** 1.5)  # Calculate the number of dashes for indentation
        spaces = "-" * const  # Create an indentation string using dashes

        if self.node_type == 'root':
        
            print(f"Node Type: {self.node_type}")  # Print the node type (root)
        
        else:
        
            print(f"|{spaces} Node Type: {self.node_type} at depth {self.depth}")  # Print node type and depth
    
        print(f"{' ' * const}   | Best Feature: {self.best_feature}")  # Print the best feature for the current node
        
        if self.best_gini is None:

            print(f"{' ' * const}   | Weighted GINI impurity of the node: {self.best_gini}")

        else:

            print(f"{' ' * const}   | Weighted GINI impurity of the node: {round(self.best_gini, 2)}")

        print(f"{' ' * const}   | Class distribution in the node: {self.class_distribution}")
        print(f"{' ' * const}   | Predicted class: {self.most_common}")   

    def print_tree(self):
        """
        Prints the entire tree starting from the current node down to the leaf nodes.
        """
        self.print_info()  # Print information about the current node
        
        if self.left is not None:
        
            self.left.print_tree()  # Recursively print the left subtree
        
        if self.right is not None:
        
            self.right.print_tree()  # Recursively print the right subtree

    def predict(self, X_test):
        
        """
        Batch prediction method to predict the class for a batch of test samples.
        """
        predictions = []

        for _, x in X_test.iterrows():
        
            values = {}
        
            for feature in X_test.columns:
        
                values.update({feature: x[feature]})

            cur_node = self  # Reset the current node for each observation

            while cur_node.depth < cur_node.max_depth:
        
                # Traverse the tree nodes all the way to the bottom
                best_feature = cur_node.best_feature

                if values.get(best_feature) == "Yes":
        
                    if cur_node.left is not None:
        
                        cur_node = cur_node.left
        
                elif values.get(best_feature) == "No":
        
                    if cur_node.right is not None:
        
                        cur_node = cur_node.right
        
                else:
        
                    break

            predictions.append(cur_node.most_common)  # Append the most common class as the prediction

        return predictions  # Return the list of predictions for all test samples




if __name__ == "__main__":

    pass