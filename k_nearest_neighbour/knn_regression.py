import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from scipy.stats import mode

class KNN_self():
    
    ############################################################################################################
    ################################### Initialization / Main methods ##########################################
    ############################################################################################################
    def __init__(self, n_neighbors : int = 3, distance_metric : str = 'euclidean', weighting : bool = False) -> None:
        
        """
        Initialize the k-Nearest Neighbors classifier.

        Parameters:
        - n_neighbors (int): Number of neighbors to consider (default is 3).
        - distance_metric (str): Distance metric used for finding neighbors (default is 'euclidean').
        - weighting (bool): If True, neighbors are weighted by inverse distance during prediction (default is False).
        """
        
        self.__check_init_method(n_neighbors, distance_metric)

        self.__n_neighbors = n_neighbors
        self.__distance_metric = distance_metric
        self.__weighting = weighting
        
    def fit(self, X_train : pd.DataFrame | pd.core.series.Series | np.ndarray, y_train : pd.DataFrame | pd.core.series.Series | np.ndarray, scaling_method : str = None) -> None:
   
        """
        Fit the k-Nearest Neighbors classifier to the training data.

        Parameters:
        - X_train (pd.DataFrame | pd.core.series.Series | np.ndarray): Training feature data.
        - y_train (pd.DataFrame | pd.core.series.Series | np.ndarray): Training labels.
        - scaling_method (str): Method for scaling the data (default is None).

        Returns:
        None
        """
        
        self.__check_fitting_method(X_train, scaling_method)
        self.__check_if_valid_data(X_train, y_train)
        
        self.__X_train = X_train.to_numpy()
        self.__y_train = y_train.to_numpy()
        self.__scaling_method = scaling_method
        
        if self.__scaling_method is not None:
            self.__X_train = self.__scale_data(self.__X_train)
            
        
    def predict(self, X_testing : pd.DataFrame | pd.core.series.Series | np.ndarray, aggregation_function : str = "mean") -> np.ndarray:
        
        """
        Make predictions using the k-Nearest Neighbors classifier.

        Parameters:
        - X_testing (pd.DataFrame | pd.core.series.Series | np.ndarray): Testing feature data.

        Returns:
        np.ndarray: Predicted classes for each data point.
        """
        
        self.__check_prediction_method(X_testing)
        self.__check_if_valid_data(X_testing)
        self.__check_if_valid_aggregation_function(aggregation_function)
        
        self.__X_test = X_testing.to_numpy()
        self.__aggregation_function = aggregation_function
        
        if self.__scaling_method is not None:
            self.__X_test = self.__scale_data(self.__X_test)

        # Get the predicted classes for each data point
        prediction = [self.__get_neighbors(point) for point in range(len(self.__X_test))]
        
        return np.asarray(prediction)

    def error(self, predictions : np.ndarray, y_testing : pd.DataFrame | pd.core.series.Series | np.ndarray, error_type : str = "MSE") -> float:
        
        """
        Calculate the error between the predictions and the actual labels.
        
        Parameters:
        - predictions (np.ndarray): Predicted classes for each data point.
        - y_testing (pd.DataFrame | pd.core.series.Series | np.ndarray): Testing labels.
        - error_type (str): Type of error to calculate (default is "MSE").
        
        Returns:
        float: Error between the predictions and the actual labels.
        """  
    
        self.__check_if_valid_data(y_testing)
        self.__check_if_valid_error_type(error_type)
        
        if error_type == "MSE":
            return np.mean(np.square(predictions - y_testing.to_numpy()))
        elif error_type == "MAE":
            return np.mean(np.abs(predictions - y_testing.to_numpy()))
        elif error_type == "RMSE":
            return np.sqrt(np.mean(np.square(predictions - y_testing.to_numpy())))
        

    ############################################################################################################
    ############################################# Scaling Data #################################################
    ############################################################################################################
    def __scale_data(self, data : np.ndarray) -> np.ndarray:

        """
        Apply scaling to the data based on the specified method.

        Parameters
        ----------
        scaling_method : str
            Scaling method to use. Options are 'min_max', 'standardization', 'mean_normalization'.

        Raises
        ------
        ValueError
            If the specified scaling method is not supported.
        """
        
        # Apply scaling based on the specified method
        if self.__scaling_method == 'min_max':
            # Min-Max scaling: scale to the range [0, 1]
            return (data - data.min()) / (data.max() - data.min())

        elif self.__scaling_method == 'standardization':
            # Standardization: scale to have mean=0 and standard deviation=1
            return (data - data.mean()) / data.std()

        elif self.__scaling_method == 'mean_normalization':
            # Mean normalization: scale to have mean=0 and range [-1, 1]
            return (data - data.mean()) / (data.max() - data.min())
    
    
    ############################################################################################################
    ########################################### Distance Methods ###############################################
    ############################################################################################################
    def __get_neighbors(self, point : np.ndarray) -> float:

        """
        Determine the most common class label among the k-nearest neighbors of a given point.

        Parameters:
        - point (np.ndarray): Data point for which neighbors are to be found.

        Returns:
        str: Most common class label among the neighbors.
        """

        distances = self.__calculate_distance(point)
            
        if self.__weighting:
            aggregation_value = self.__get_weighted_prediction(distances)

        else:
            aggregation_value = self.__get_prediction(distances)
            
        return aggregation_value
        
    def __calculate_distance(self, point : np.ndarray) -> np.ndarray:
        
        """
        Calculate distances between a given data point and all training data points.

        Parameters:
        - point (np.ndarray): Data point for which distances are to be calculated.

        Returns:
        np.ndarray: Array containing distances, indices, and weights for each training data point.
        """
        
        # Compute the distance between the new data point and each training data point
        distances = self.__get_distance(point)
        weights = [1.0] * len(distances) # Default weight is 1.0
        
        # Apply weighting (inverse of distance)
        if self.__weighting:
            weights = 1.0 / (distances + 1e-6) # Adding a small constant to avoid division by zero
        
        # stack the distances, indices and weights together for sorting
        return np.column_stack((distances, np.argsort(distances), weights))

    def __get_weighted_prediction(self, distances : np.ndarray) -> float:
        
        """
        Determine the weighted prediction based on the k-nearest neighbors.
        
        Parameters:
        - distances (np.ndarray): Array containing distances, indices, and weights for each training data point.
        
        Returns:
        float: Weighted prediction based on the k-nearest neighbors.
        """        
        
        # Sort the distances in descending order
        distances = sorted(distances, key=lambda x: x[2], reverse = True)
    
        # Get the k-nearest neighbors and their corresponding weights
        nearest_neighbors = distances[:self.__n_neighbors]

        # Extract neighbor values and weights
        neighbor_values = [self.__y_train[int(index)] for _, index, _ in nearest_neighbors] # index is a float, so we need to convert it to an int
        weights = [weight for _, _, weight in nearest_neighbors]
            
        # Calculate the weighted prediction based on the chosen aggregation function
        if self.__aggregation_function == "mean":
            total_weight = sum(weights)
            weighted_average = sum(value * weight for value, weight in zip(neighbor_values, weights)) / total_weight if total_weight != 0 else 0.0
            return weighted_average

        elif self.__aggregation_function == "median":
            # Calculate the cumulative weights
            cumulative_weights = np.cumsum(weights)
            total_weight = cumulative_weights[-1]

            # Find the index of the median in the sorted cumulative weights
            median_index = np.searchsorted(cumulative_weights, total_weight / 2)

            # Use the corresponding neighbor value as the weighted median
            weighted_median = neighbor_values[median_index]
            return weighted_median

        elif self.__aggregation_function == "mode":
            # Calculate the mode considering weights
            weighted_mode, _ = mode(neighbor_values, weights=weights)
            return weighted_mode[0]
    
    def __get_prediction(self, distances : np.ndarray) -> float:
        
        """
        Determine the prediction based on the k-nearest neighbors.
        
        Parameters:
        - distances (np.ndarray): Array containing distances, indices, and weights for each training data point.
        
        Returns:
        float: Prediction based on the k-nearest neighbors.        
        """
    
        distances = sorted(distances, key=lambda x: x[0])
        
        # get the k-nearest neighbors
        nearest_neighbors = distances[:self.__n_neighbors]
        
        # extract neighbor values
        neighbor_values = [self.__y_train[int(index)] for _, index, _ in nearest_neighbors] # index is a float, so we need to convert it to an int
        
        # calculate the prediction based on the chosen aggregation function
        if self.__aggregation_function == "mean":
            return np.mean(neighbor_values)
        elif self.__aggregation_function == "median":
            return np.median(neighbor_values)
        elif self.__aggregation_function == "mode":
            return Counter(neighbor_values).most_common(1)[0][0]
        
    def __get_distance(self, point : np.ndarray) -> np.ndarray:
        
        """
        Calculate the distance between data points of training data to new data point based on the specified metric.

        Parameters
        ----------
        points : np.ndarray
            Array containing the data points.

        Returns
        -------
        np.ndarray
            Array containing the calculated distances.

        Raises
        ------
        ValueError
            If the specified distance metric is not supported.
        """
                
        if self.__distance_metric == 'euclidean':
            # Euclidean distance: square root of the sum of squared differences
            return np.sqrt(np.sum(np.square(point - self.__X_train), axis=1))
        
        elif self.__distance_metric == 'manhattan':
            # Manhattan distance: sum of absolute differences
            return np.sum(np.abs(point - self.__X_train), axis=1)
        
        elif self.__distance_metric == 'squared_euclidean':
            # Squared Euclidean distance: sum of squared differences
            return np.sum(np.square(point - self.__X_train), axis=1)
        
        elif self.__distance_metric == 'canberra':
            # Canberra distance: sum of absolute differences normalized by the sum of absolute values
            return np.sum(np.abs(point - self.__X_train) / (np.abs(point) + np.abs(self.__X_train)), axis=1)
    
    
    ############################################################################################################
    ########################################### Validating Data ################################################
    ############################################################################################################
    def __check_if_valid_data(self, X_data : pd.DataFrame | pd.core.series.Series | np.ndarray, y_data : pd.DataFrame | pd.core.series.Series | np.ndarray = None) -> None:
        
        """
        Check if the input data is a valid type.

        Parameters
        ----------
        X_data : pd.DataFrame | pd.core.series.Series | np.ndarray
            The input feature data to be checked.
        y_data : pd.DataFrame | pd.core.series.Series | np.ndarray, optional
            The input label data to be checked (default is None).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the feature data is not a pandas DataFrame or a numpy array,
            or if the label data is not a pandas DataFrame or a numpy array.
        """
        
        # Check if data is a valid type
        if not isinstance(X_data, (pd.core.frame.DataFrame, pd.core.series.Series, np.ndarray)):
            raise TypeError("X_Data must be a pandas DataFrame or a numpy array")

        if not isinstance(y_data, (pd.core.frame.DataFrame, pd.core.series.Series, np.ndarray)) and y_data is not None:
            raise TypeError("Y_Data must be a pandas DataFrame or a numpy array")
    
    def __check_init_method(self, n_neighbors : int, distance_metric : str) -> None:
        
        """
        Check the validity of parameters used in the initialization method.

        Parameters
        ----------
        n_neighbors : int
            Number of neighbors to consider.
        distance_metric : str
            Distance metric used for finding neighbors.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the number of neighbors is not greater than 0,
            or if the distance metric is not supported.
        """
        
        if n_neighbors <= 0:
            raise ValueError(f"Number of neighbors {n_neighbors} must be larger than 0")
        
        if distance_metric not in ['euclidean', 'manhattan', 'squared_euclidean', 'canberra']:
            raise ValueError(f"Distance metric {distance_metric} not supported (supported metrics: euclidean, manhattan, squared_euclidean, canberra)")
        
    def __check_fitting_method(self, X_data : pd.DataFrame | pd.core.series.Series | np.ndarray, scaling_method : str) -> None:
        
        """
        Check the validity of parameters used in the fitting method.

        Parameters
        ----------
        X_data : pd.DataFrame | pd.core.series.Series | np.ndarray
            Training feature data.
        scaling_method : str
            Method for scaling the data.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the number of neighbors is greater than the number of data points,
            or if the scaling method is not supported.
        """
        
        if self.__n_neighbors > len(X_data):
            raise ValueError(f"Number of neighbors {self.__n_neighbors} cannot be larger than the number of data points {len(X_data)}")
        
        if scaling_method is not None and scaling_method not in ['min_max', 'standardization', 'mean_normalization']:
            raise ValueError(f"Unsupported scaling method: {scaling_method}, supported methods are: min_max_normalization, standardization, mean_normalization")
        
    def __check_prediction_method(self, data: pd.DataFrame | pd.core.series.Series | np.ndarray) -> None:
        
        """
        Check the validity of parameters used in the prediction method.

        Parameters
        ----------
        data : pd.DataFrame | pd.core.series.Series | np.ndarray
            Testing feature data.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the model has not been fitted with data,
            or if the number of features in the testing data is not equal to the number of features in the training data.
        """
        
        if self.__X_train is None:
            raise AttributeError("Model needs to be fitted first with data!")

        if data.shape[1] != self.__X_train.shape[1] and data.shape[1] != self.__X_train.shape[1] + 1:
            raise ValueError(f"Number of features in data {data.shape[1]} must be equal to the number of features in training data {self.__X_train.shape[1]}")        
    
    def __check_if_valid_aggregation_function(self, aggregation_function : str) -> None:
        
        """
        Check the validity of the aggregation function.

        Parameters
        ----------
        aggregation_function : str
            Aggregation function to use.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the aggregation function is not supported.
        """
        
        if aggregation_function not in ["mean", "mode", "median"]:
            raise ValueError(f"Aggregation function {aggregation_function} not supported (supported functions: mean, mode, median)")
        
    def __check_if_valid_error_type(self, error_type : str) -> None:
        
        """
        Check the validity of the error type.

        Parameters
        ----------
        error_type : str
            Error type to use.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the error type is not supported.
        """
        
        if error_type not in ["MSE", "MAE", "RMSE"]:
            raise ValueError(f"Error type {error_type} not supported (supported types: MSE, MAE, RMSE)")
        
        
############################################################################################################
############################################# Testing ######################################################
############################################################################################################
if __name__ == "__main__":
    
    #read in the data using pandas
    df = pd.read_csv('../datasets/diabetes.csv')
    df = df.reindex(np.random.permutation(df.index)).reset_index(drop = True)
    
    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]

    #split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    knn_example = KNN_self(n_neighbors = 2, distance_metric="euclidean", weighting=False)
    knn_example.fit(X_train, y_train)
    
    predictions = knn_example.predict(X_test)
    print("Predictions: ", predictions)
    print("Accuracy: ", knn_example.accuracy(predictions, y_test))