import pandas as pd
import numpy as np
import sys
from collections import Counter


# Feature scaling (standardization)
def scale_features(X):

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_scaled = (X - mean) / std
    
    return X_scaled

############################################################################################################

# create custom train test function
def custom_train_test_split(X, y, training_size=0.8, random_state=None, num_shuffles=100):

    # Check if X and y have the same number of rows
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")
    
    # Set a seed if one is defined
    if random_state is None:
        random_state = 42
    
    # Combine X and y into a single DataFrame
    df = pd.concat([X, y], axis=1)
    
    # Check if shuffle is requested
    if num_shuffles <= 0:
        # Determine the split index
        split_index = int(len(df) * training_size)
        
        # Split the DataFrame into training and testing sets
        train_df = df.iloc[:split_index]
        test_df = df.iloc[split_index:]
    
    else:
    
        # Shuffle the DataFrame for a specified number of times
        np.random.seed(random_state)  # Set seed for reproducibility
        shuffled_indices = np.arange(len(df))
    
        for _ in range(num_shuffles):
    
            np.random.shuffle(shuffled_indices)
    
        shuffled_df = df.iloc[shuffled_indices].reset_index(drop=True)
        
        # Determine the split index after shuffling
        split_index = int(len(shuffled_df) * training_size)
        
        # Split the shuffled DataFrame into training and testing sets
        train_df = shuffled_df.iloc[:split_index]
        test_df = shuffled_df.iloc[split_index:]
    
    # Separate features and target variable in the training and testing sets
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    
    return X_train, X_test, y_train, y_test

############################################################################################################

def accuracy_score(predictions, y_test):

    right_prediction = 0

    for i in predictions:

        if predictions[i] == y_test.iloc[i]:

            right_prediction += 1

    
    return ( right_prediction / len(y_test) ) * 100

############################################################################################################

class KNN_self():

    def __init__(self, n_neighbors = 3, distanceMetric = 'euclidean', weighting = False):
       
        self.n_neighbors = n_neighbors
        self.distanceMetric = distanceMetric
        self.weighting = weighting

    def euclidean_distance(self, X_test):
        
        distances = []

        for index in range(len(self.X_train)):
            
            # Compute the Euclidean distance
            distance = np.sqrt(np.sum(np.square(X_test - self.X_train.iloc[[index]].to_numpy())))

            # Apply weighting (inverse of distance)
            if self.weighting:
                
                weight = 1.0 / (distance + 1e-6)  # Adding a small constant to avoid division by zero
                distances.append([distance, self.X_train.iloc[[index]].index[0], weight])
            
            else:
                distances.append([distance, self.X_train.iloc[[index]].index[0]])

        return distances

    
    def fit(self, X_train, y_train):
   
        assert len(X_train) == len(y_train)
        self.X_train = X_train
        self.y_train = y_train

    def get_neighbors(self, X_test):

        distances = self.euclidean_distance(X_test)
            
        if self.weighting:
            
            distances = sorted(distances, key=lambda x: x[2], reverse = True)
        
            # Get the k-nearest neighbors and their corresponding weights
            nearest_neighbors = distances[:self.n_neighbors]

            # Calculate weighted class probabilities
            class_probabilities = {}
            for distance, index, weight in nearest_neighbors:

                neighbor_class = self.y_train[self.y_train.index == index].values[0]
                
                if neighbor_class in class_probabilities:
                
                    class_probabilities[neighbor_class] += weight
                
                else:
                    class_probabilities[neighbor_class] = weight

            # Make a prediction based on the class with the highest weighted probability
            most_common_neighbor = max(class_probabilities, key=class_probabilities.get)

            return most_common_neighbor

        else:

            neighbors = []
            distances = sorted(distances, key=lambda x: x[0])

            # make a list of the k neighbors' targets
            for i in range(self.n_neighbors):
                
                index = distances[i][1]
                neighbors.append(self.y_train[self.y_train.index == index].values[0])

            # return most common target
            most_common_neighbor = Counter(neighbors).most_common(1)[0][0]

            return most_common_neighbor


    def predict(self, X_test):

        predictions = []

        # loop over all observations
        for i in range(len(X_test)):

            predictions.append(self.get_neighbors(X_test.iloc[[i]].to_numpy()))

        return np.asarray(predictions)
    
############################################################################################################

class KFold_self:
    
    def __init__(self, n_splits=5, shuffle=False):

        self.n_splits = n_splits
        self.shuffle = shuffle

    def split(self, X):
        
        indices = np.arange(len(X))
        fold_size = len(X) // self.n_splits
        remainder = len(X) % self.n_splits

        for i in range(self.n_splits):
        
            if self.shuffle:
            
                np.random.seed(np.random.randint(0, 100))
                np.random.shuffle(indices)
        
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.n_splits - 1 else (i + 1) * fold_size + remainder

            test_indices = indices[start:end]
            train_indices = np.concatenate((indices[:start], indices[end:]))

            yield sorted(train_indices), sorted(test_indices)


############################################################################################################
############################################################################################################
############################################################################################################

if __name__ == "__main__":

    #read in the data using pandas
    df = pd.read_csv('diabetes.csv') #https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
    df = df.reindex(np.random.permutation(df.index)).reset_index(drop = True)
    print(df)

    #-------------------------------------------------------------------------------------------------------------------------------------------

    X = df.drop('Outcome', axis = 1)
    y = df['Outcome']

    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, training_size = 0.85, random_state = 42)
    # Normal test set
    test_set = pd.merge(X_test, y_test, left_index=True, right_index=True)


    X_scaled = scale_features(X)
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = custom_train_test_split(X_scaled, y, training_size = 0.85, random_state = 42)
    # Scaled normal test set
    test_set_scaled = pd.merge(X_test_scaled, y_test_scaled, left_index=True, right_index=True)

    #-------------------------------------------------------------------------------------------------------------------------------------------

    kfold = KFold_self(n_splits=10, shuffle=True)

    #-------------------------------------------------------------------------------------------------------------------------------------------
    input_k = sys.argv[1]

    if input_k.isdigit():

        input_k = int(input_k)

    else:
        exit()

    k_values = range(1, input_k, 2)
    mean_accuracy_scores = {'K_Value' : [], 'Mean_Accuracy_Score' : [], 'Accuracy_Scores' : [], 'Weighted' : [], 'Scaled' : []}

    for k in k_values:

        for weighted in [True, False]:

            accuracy_scores = []

            for train_index, test_index in kfold.split(X_train):

                X_train_kfold, X_validation = X_train[X_train.index.isin(train_index)], X_train[X_train.index.isin(test_index)]
                y_train_kfold, y_validation = y_train[y_train.index.isin(train_index)], y_train[y_train.index.isin(test_index)]

                knn = KNN_self(n_neighbors = k, weighting=weighted)

                knn.fit(X_train_kfold, y_train_kfold)

                predictions = knn.predict(X_validation)

                accuracy = accuracy_score(predictions, y_validation)

                accuracy_scores.append(accuracy)

            mean_accuracy_scores['K_Value'].append(k)
            mean_accuracy_scores['Mean_Accuracy_Score'].append(np.mean(accuracy_scores))
            mean_accuracy_scores['Accuracy_Scores'].append(accuracy_scores)
            mean_accuracy_scores['Weighted'].append(weighted)
            mean_accuracy_scores['Scaled'].append(False)

        for weighted in [True, False]:

            accuracy_scores = []

            for train_index, test_index in kfold.split(X_train_scaled):

                X_train_kfold_scaled, X_validation_scaled = X_train_scaled[X_train_scaled.index.isin(train_index)], X_train_scaled[X_train_scaled.index.isin(test_index)]
                y_train_kfold_scaled, y_validation_scaled = y_train_scaled[y_train_scaled.index.isin(train_index)], y_train_scaled[y_train_scaled.index.isin(test_index)]

                knn = KNN_self(n_neighbors = k, weighting=weighted)

                knn.fit(X_train_kfold_scaled, y_train_kfold_scaled)

                predictions = knn.predict(X_validation_scaled)

                accuracy = accuracy_score(predictions, y_validation_scaled)

                accuracy_scores.append(accuracy)

            mean_accuracy_scores['K_Value'].append(k)
            mean_accuracy_scores['Mean_Accuracy_Score'].append(np.mean(accuracy_scores))
            mean_accuracy_scores['Accuracy_Scores'].append(accuracy_scores)
            mean_accuracy_scores['Weighted'].append(weighted)
            mean_accuracy_scores['Scaled'].append(True)


    #-------------------------------------------------------------------------------------------------------------------------------------------

    df = pd.DataFrame.from_dict(mean_accuracy_scores)
    # df.to_csv("knn_mean_accuracy_scores.csv", index = False)

    print("\n\n")
    print(df)

    #-------------------------------------------------------------------------------------------------------------------------------------------

    best_k_score = df['Mean_Accuracy_Score'].max()
    best_k = df[df['Mean_Accuracy_Score'] == best_k_score]['K_Value'].values[0]
    best_weighted = df[df['Mean_Accuracy_Score'] == best_k_score]['Weighted'].values[0]
    best_scaled = df[df['Mean_Accuracy_Score'] == best_k_score]['Scaled'].values[0]

    print("\n\n")
    print(f"Best K = {best_k} for training/validation with an average accuracy score of {best_k_score:.4f}% with scaled = {best_scaled} and weighted = {best_weighted}")
    print("\n\n")


    #-------------------------------------------------------------------------------------------------------------------------------------------
    
    knn_testing = KNN_self(n_neighbors = best_k, weighting=weighted)

    if best_scaled:
        
        knn_testing.fit(X_train_scaled, y_train_scaled)
        predictions = knn.predict(X_test_scaled)
        accuracy_testing = accuracy_score(predictions, y_test_scaled)

    else:

        knn_testing.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        accuracy_testing = accuracy_score(predictions, y_test)

    
    print(f"Best K = {best_k} for testing with an accuracy score of {accuracy_testing:.4f}%")
    print("\n\n")








