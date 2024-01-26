import pandas as pd
import numpy as np

class train_test_split():
    
    """
    A class for custom train-test splitting of data.

    Parameters:
    - *data : pd.DataFrame | np.ndarray | pd.core.series.Series
        The input data to be split into training and testing sets.
    - training_size : float | int, default=None
        The proportion or absolute number of samples to include in the training set.
    - testing_size : float | int, default=None
        The proportion or absolute number of samples to include in the testing set.
    - random_state : int, default=None
        Seed for random number generation to ensure reproducibility.
    - num_shuffles : int, default=None
        Number of shuffles to perform on the data before splitting.

    Methods:
    - __call__():
        Split the data into training and testing sets based on the specified parameters.
        Returns the split data as pandas DataFrames.
    """
    
    def __init__(self, *data : pd.DataFrame | np.ndarray | pd.core.series.Series, training_size : float | int = None, testing_size : float | int = None, random_state : int = None, num_shuffles : int = None):
    
        """
        Initialize the train_test_split instance.

        Parameters:
        - *data : pd.DataFrame | np.ndarray | pd.core.series.Series
            The input data to be split into training and testing sets.
        - training_size : float | int, default=None
            The proportion or absolute number of samples to include in the training set.
        - testing_size : float | int, default=None
            The proportion or absolute number of samples to include in the testing set.
        - random_state : int, default=None
            Seed for random number generation to ensure reproducibility.
        - num_shuffles : int, default=None
            Number of shuffles to perform on the data before splitting.
        """
    
        # checks
        self.__n_arrays, self.__type_data, self.__column_names = self.__check_data(data)
        self.__check_training_testing_size(training_size, testing_size)
        
        # set variables
        self.__set_length_data(data)
        self.__transform_data(data)
        self.__set_training_testing_size(training_size, testing_size)
    
        self.__random_state = random_state
        self.__num_shuffles = num_shuffles
            
    def __call__(self):
        
        # split data
        array = self.__split_data()
        
        # convert back to pd.DataFrame
        for index, _type in enumerate(self.__type_data):

            if _type == pd.DataFrame or _type == pd.core.series.Series:
                
                for i in range(2):
                    
                    if index > 0: i += 1

                    array[index + i] = pd.DataFrame(array[index + i])
                    array[index + i].columns = self.__column_names[index]

        if len(array) == 2:
            return array[0], array[1]
        else:
            return array[0], array[1], array[2], array[3]
    
    
    def __split_data(self) -> list:
        
        """
        Split the data into training and testing sets based on the number of input arrays.

        Returns:
        list: List containing the split data arrays.
        """
        
        if self.__n_arrays == 1:
            array = self.__split_data_1_array()
        else:
            array = self.__split_data_2_arrays()
            
        return array
    
    def __split_data_1_array(self) -> list:
        
        """
        Split the data when only one array is provided.

        Returns:
        list: List containing the split data arrays.
        """
        
        # set random seed
        self.__set_random_state()
        
        # shuffle data
        if self.__num_shuffles is not None:
            for _ in range(self.__num_shuffles):
                np.random.shuffle(self.__data[0])
                
        training_indices, testing_indices = self.__get_random_indices()
        
        # split data into training and testing set and return
        return [ self.__data[0][training_indices], self.__data[0][testing_indices] ]
        
    def __split_data_2_arrays(self) -> list:
        
        """
        Split the data when two arrays are provided.

        Returns:
        list: List containing the split data arrays.
        """
        
        # set random seed
        self.__set_random_state()
        
        # shuffle data
        if self.__num_shuffles is not None:
            for i in range(self.__num_shuffles):
                np.random.shuffle(self.__data[0])
                np.random.shuffle(self.__data[1])
                
        training_indices, testing_indices = self.__get_random_indices()
        
        # split data into training and testing set and return
        return [ self.__data[0][training_indices], self.__data[0][testing_indices], 
                 self.__data[1][training_indices], self.__data[1][testing_indices] ]
        
    def __get_random_indices(self) -> tuple:
        
        """
        Generate random indices for training and testing sets.

        Returns:
        tuple: Tuple containing training and testing indices.
        """
        
        training_indices = np.random.choice(self.__length_data, size = int(self.__training_size * self.__length_data), replace = False)
        testing_indices = np.setdiff1d(np.arange(self.__length_data), training_indices)
        
        return (training_indices, testing_indices)
    
    def __set_random_state(self) -> None:
        
        """
        Set the random seed for reproducibility.
        """
        
        if self.__random_state is not None:
            np.random.seed(self.__random_state)
    
    ############################################################################################################
    ########################################### Set Variables ##################################################
    ############################################################################################################  
    def __transform_data(self, data: pd.DataFrame | np.ndarray | pd.core.series.Series) -> None:
        
        """
        Transform input data into a consistent format (np.ndarray) for further processing.

        Parameters:
        - data : pd.DataFrame | np.ndarray | pd.core.series.Series
            The input data to be transformed.
        """
        
        if self.__n_arrays == 1:
            if isinstance(data[0], pd.DataFrame):
                # convert to np.ndarray
                self.__data = [data[0].to_numpy()]
            elif isinstance(data[0], pd.core.series.Series):
                # convert to np.ndarray
                self.__data = [data[0].values]
            else:
                self.__data = [data[0]]

        else:
            if isinstance(data[0], pd.DataFrame) or isinstance(data[0], pd.core.series.Series):
                # convert to np.ndarray
                data_1 = data[0].to_numpy()
            else:
                data_1 = data[0]

            if isinstance(data[1], pd.DataFrame) or isinstance(data[1], pd.core.series.Series):
                # convert to np.ndarray
                data_2 = data[1].to_numpy()
            else:
                data_2 = data[1]

            self.__data = [data_1, data_2]       
    
    def __set_length_data(self, data : pd.DataFrame | np.ndarray | list) -> None:
        
        """
        Set the length of the input data.

        Parameters:
        - data : pd.DataFrame | np.ndarray | list
            The input data.
        """
        
        self.__length_data = len(data[0])

        
    def __set_training_testing_size(self, training_size : float | int | None, testing_size : float | int | None) -> None:
        
        """
        Set the training and testing sizes based on user inputs.

        Parameters:
        - training_size : float | int | None
            The specified training size.
        - testing_size : float | int | None
            The specified testing size.
        """
        
        if training_size is None and testing_size is not None and type(testing_size) == float:
            self.__training_size = 1 - testing_size
        
        elif training_size is None and testing_size is not None and type(testing_size) == int:
            self.__training_size = self.__length_data - testing_size
            
        elif training_size is not None and type(training_size) == float and testing_size is None:
            self.__training_size = training_size
            
        elif training_size is not None and type(training_size) == int and testing_size is None:
            self.__training_size = training_size

        elif training_size is not None and testing_size is not None:
            self.__training_size = 0.75


    ############################################################################################################
    ########################################### Validating Data ################################################
    ############################################################################################################
    def __check_data(self, data : pd.DataFrame | np.ndarray | pd.core.series.Series) -> int:
    
        """
        Check the validity of the input data.

        Parameters:
        - data : pd.DataFrame | np.ndarray | pd.core.series.Series
            The input data.

        Returns:
        tuple: Tuple containing the number of arrays, types of arrays, and column names.
        """
    
        n_arrays = len(data)
            
        if n_arrays == 0:
            raise ValueError("At least one array required as input")
        
        elif n_arrays > 2:
            raise ValueError("Only two arrays are max. allowed as input")

        # check if all arrays have the same length
        lengths = [len(array) for array in data]
        
        if len(set(lengths)) != 1:
            raise ValueError("All arrays must have the same length")

        # check if all arrays are either pd.DataFrame or np.ndarray
        modified_data = []
        for array in data:
            if not isinstance(array, (pd.DataFrame, np.ndarray, pd.core.series.Series)):
                raise TypeError("All arrays must be either pd.DataFrame or np.ndarray or pd.Series")

            if isinstance(array, pd.core.series.Series):
                array = array.to_frame()
            
            modified_data.append(array)

        # check if all arrays are either pd.DataFrame or np.ndarray
        type_data = [type(array) for array in modified_data]
        
        # check if all arrays are either pd.DataFrame or np.ndarray
        column_names = []
        for index, _type in enumerate(type_data):
            if _type == pd.DataFrame or _type == pd.core.series.Series:
                column_names.append(modified_data[index].columns)
            else:
                column_names.append(None)
                        
        return n_arrays, type_data, column_names
        
    
    def __check_training_testing_size(self, training_size : float | int | None, testing_size : float | int | None) -> None:
        
        """
        Check the validity of training and testing sizes.

        Parameters:
        - training_size : float | int | None
            The specified training size.
        - testing_size : float | int | None
            The specified testing size.
        """
        
        if type(training_size) == float and (training_size < 0 or training_size > 1):
            raise ValueError("training_size must be between 0 and 1")
    
        if type(testing_size) == float and (testing_size < 0 or testing_size > 1):
            raise ValueError("testing_size must be between 0 and 1")
        
        if type(training_size) == int and training_size < 0:
            raise ValueError("training_size must be greater than 0 in order to include at least one sample in the training set")
        
        if type(testing_size) == int and testing_size < 0:
            raise ValueError("testing_size must be greater than 0 in order to include at least one sample in the testing set")
        
        if type(training_size) == int and type(testing_size) == int and training_size + testing_size > len(self.__length_data):
            raise ValueError("training_size + testing_size cannot be greater than the number of samples in the dataset")
        
        if type(training_size) == float and type(testing_size) == float and training_size + testing_size > 1:
            raise ValueError("training_size + testing_size cannot be greater than 1")
