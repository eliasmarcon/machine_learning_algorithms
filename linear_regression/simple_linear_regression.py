import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split


class SimpleLinearRegression:
    """
    SimpleLinearRegression class for fitting and predicting using simple linear regression.

    Parameters
    ----------
    fit_intercept : bool, optional
        Indicates whether to fit an intercept term in the model. The default is True.

    Attributes
    ----------
    _slope : float
        The slope of the regression line.
    _intercept : float
        The intercept of the regression line.
    __X_train : numpy.ndarray
        The training data for the independent variable.
    __y_train : numpy.ndarray
        The training data for the dependent variable.
    __X_test : numpy.ndarray
        The testing data for the independent variable.
    __y_pred : numpy.ndarray
        The predicted values for the dependent variable.
    __y_test : numpy.ndarray
        The true values for the dependent variable in testing data.
    __x_train_col_name : str
        The name of the column in the training data used for the independent variable.
    __y_train_col_name : str
        The name of the column in the training data used for the dependent variable.

    Methods
    -------
    fit(X_train, y_train, sample_weights=None)
        Fit the simple linear regression model to the training data.

    predict(X_test)
        Predict the dependent variable for new data.

    get_metric(y_pred, y_test, metric)
        Calculate a specified metric between predicted and true values.

    plot_scatter(title=None, xlabel=None, ylabel=None)
        Plot a scatter plot of the training data.

    plot_regression_line(title=None, xlabel=None, ylabel=None)
        Plot the regression line on top of the scatter plot of the training data.

    plot_residuals(title=None, xlabel=None, ylabel=None)
        Plot the residuals of the model.

    plot_testing_and_prediction(title=None, xlabel=None, ylabel=None)
        Plot the training data, fitted regression line, and testing data with predictions.

    """
    
    def __init__(self, fit_intercept : bool = True):
        
        """
        Initialize the SimpleLinearRegression model.

        Parameters
        ----------
        fit_intercept : bool, optional
            Indicates whether to fit an intercept term in the model. The default is True.

        Returns
        -------
        None
        """
        
        self.__fit_intercept = fit_intercept
        
    def fit(self, X_train: pd.core.frame.DataFrame | pd.core.series.Series | np.ndarray, y_train: pd.core.frame.DataFrame | pd.core.series.Series | np.ndarray, sample_weights: np.ndarray | list = None) -> None:
        """
        Fit the simple linear regression model to the training data.

        Parameters
        ----------
        X_train : pandas.DataFrame, pandas.Series, numpy.ndarray
            Training data for the independent variable.
        y_train : pandas.DataFrame, pandas.Series, numpy.ndarray
            Training data for the dependent variable.
        sample_weights : numpy.ndarray or list, optional
            Sample weights for the training data. The default is None.

        Returns
        -------
        None
        """

        self.__check_if_valid_data(X_train, y_train, sample_weights)

        self.__X_train, self.__x_train_col_name = self.__check_X_train(X_train)
        self.__y_train, self.__y_train_col_name = self.__check_y_train(y_train)

        sample_weights = self.__check_sample_weights(sample_weights)
        
        self.__calculate_coefficients(sample_weights)
        
    
    def __check_sample_weights(self, sample_weights : np.ndarray | list) -> np.ndarray:
        
        """
        Check and convert sample weights.
        
        Parameters
        ----------
        sample_weights : numpy.ndarray or list, optional
            Sample weights for the training data. The default is None.
        
        Returns
        -------
        numpy.ndarray
            Converted sample weights.
        """
        
        if sample_weights is None:
            return np.ones_like(self.__X_train)
        
        if len(sample_weights) != len(self.__X_train):
            raise ValueError("Length of sample weights must be equal to the length of X_train")
        
        if isinstance(sample_weights, list):
            return np.array(sample_weights)
        else:
            return sample_weights    
    
        
    def predict(self, X_test : pd.core.frame.DataFrame | pd.core.series.Series | np.ndarray) -> np.ndarray:
        
        """
        Predict the dependent variable for new data.

        Parameters
        ----------
        X_test : pandas.DataFrame, pandas.Series, numpy.ndarray
            New data for the independent variable.

        Returns
        -------
        numpy.ndarray
            Predicted values for the dependent variable.
        """
        
        self.__X_test = self.__check_X_test(X_test)
        
        return self._intercept + self._slope * self.__X_test 


    ############################################################################################################
    ########################################### Coefficients ###################################################
    ############################################################################################################    
    def __calculate_coefficients(self, sample_weights: np.ndarray) -> None:
        """
        Calculate coefficients for the linear regression model.

        Parameters
        ----------
        sample_weights : numpy.ndarray
            Sample weights for the training data.

        Returns
        -------
        None
        """
        if self.__fit_intercept:
            self.__calculate_coefficients_with_intercept(sample_weights)
        else:
            self.__calculate_coefficients_without_intercept(sample_weights)

    def __calculate_coefficients_with_intercept(self, sample_weights: np.ndarray) -> None:
        """
        Calculate coefficients with intercept for the linear regression model.

        Parameters
        ----------
        sample_weights : numpy.ndarray
            Sample weights for the training data.

        Returns
        -------
        None
        """
        numerator = np.sum(sample_weights * (self.__X_train - np.mean(self.__X_train)) * (self.__y_train - np.mean(self.__y_train)))
        denominator = np.sum(sample_weights * (self.__X_train - np.mean(self.__X_train)) ** 2)

        self._slope = numerator / denominator
        self._intercept = np.mean(self.__y_train) - self._slope * np.mean(self.__X_train)

    def __calculate_coefficients_without_intercept(self, sample_weights: np.ndarray) -> None:
        """
        Calculate coefficients without intercept for the linear regression model.

        Parameters
        ----------
        sample_weights : numpy.ndarray
            Sample weights for the training data.

        Returns
        -------
        None
        """
        numerator = np.sum(sample_weights * self.__X_train * self.__y_train)
        denominator = np.sum(sample_weights * self.__X_train ** 2)

        self._slope = numerator / denominator
        self._intercept = 0
        
        
    ############################################################################################################
    ######################################## Calculating Metrics ###############################################
    ############################################################################################################
    def get_metric(self, y_pred : pd.core.frame.DataFrame | np.ndarray | pd.core.series.Series, y_test : pd.core.frame.DataFrame | np.ndarray | pd.core.series.Series, metric : str):
        
        """
        Calculate a specified metric between predicted and true values.

        Parameters
        ----------
        y_pred : pandas.DataFrame, pandas.Series, numpy.ndarray
            Predicted values for the dependent variable.
        y_test : pandas.DataFrame, pandas.Series, numpy.ndarray
            True values for the dependent variable.
        metric : str
            The metric to calculate ('mean_squared_error', 'R_squared', 'MSE', or 'R2').

        Returns
        -------
        float
            The calculated metric.
        """
        
        self.__check_metric(metric)
        self.__y_pred, self.__y_test = self.__check_prediction_testing(y_pred, y_test)
        
        return self.__get_metric(metric)
    
    def __get_metric(self, metric):
        
        if metric == 'mean_squared_error' or metric == 'MSE':
            return np.mean((self.__y_test - self.__y_pred) ** 2)
        
        elif metric == 'R_squared' or metric == 'R2':
            return 1 - (np.sum((self.__y_test - self.__y_pred) ** 2) / np.sum((self.__y_test - np.mean(self.__y_test)) ** 2))  
      
        
    ############################################################################################################
    ########################################### Plotting Data ##################################################
    ############################################################################################################    
    def plot_scatter(self, title : str = None, xlabel : str = None, ylabel : str = None) -> None:
        
        """
        Plot a scatter plot of the training data.

        Parameters
        ----------
        title : str, optional
            Title of the plot. The default is None.
        xlabel : str, optional
            Label for the x-axis. The default is None.
        ylabel : str, optional
            Label for the y-axis. The default is None.

        Returns
        -------
        None
        """
        
        title, xlabel, ylabel = self.__get_plot_attributes(title, xlabel, ylabel)
        
        self.__standard_figure_size()
        plt.scatter(self.__X_train, self.__y_train, color = 'skyblue', label='Data Points')
        self.__plot_attributes(title, xlabel, ylabel)

    def plot_regression_line(self, title : str = None, xlabel : str = None, ylabel : str = None) -> None:
        
        """
        Plot the regression line on top of the scatter plot of the training data.

        Parameters
        ----------
        title : str, optional
            Title of the plot. The default is None.
        xlabel : str, optional
            Label for the x-axis. The default is None.
        ylabel : str, optional
            Label for the y-axis. The default is None.

        Returns
        -------
        None
        """
        
        title, xlabel, ylabel = self.__get_plot_attributes(title, xlabel, ylabel)
        
        self.__standard_figure_size()
        plt.scatter(self.__X_train, self.__y_train, color = 'skyblue', label='Data Points')
        plt.plot(self.__X_train, self._intercept + self._slope * self.__X_train, color = 'red', label='Fitted Regression Line')
        self.__plot_attributes(title, xlabel, ylabel)
        
    def plot_residuals(self, title : str = None, xlabel : str = None, both : bool = False) -> None:
        
        """
        Plot the residuals of the model.

        Parameters
        ----------
        title : str, optional
            Title of the plot. The default is None.
        xlabel : str, optional
            Label for the x-axis. The default is None.
        ylabel : str, optional
            Label for the y-axis. The default is None.

        Returns
        -------
        None
        """
        
        if xlabel is None:
            xlabel = "y_train - y_train_pred"
            
        y_train_pred = self._intercept + self._slope * self.__X_train # Predicted training values for the dependent variable
        self._residuals = self.__y_train - y_train_pred # Residuals
        
        self.__standard_figure_size()
        sns.histplot(self._residuals, bins = 15, kde=True)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Density")
        plt.show()
        
        
        if both:
            self.__standard_figure_size()
            plt.scatter(self.__X_train, self._residuals, label = "Residuals vs. X_train", color = "blue")
            self.__plot_attributes("Residuals", self.__x_train_col_name, "Residuals")
            
    def plot_testing_and_prediction(self, title : str = None, xlabel : str = None, ylabel : str = None) -> None:
        
        """
        Plot the training data, fitted regression line, and testing data with predictions.

        Parameters
        ----------
        title : str, optional
            Title of the plot. The default is None.
        xlabel : str, optional
            Label for the x-axis. The default is None.
        ylabel : str, optional
            Label for the y-axis. The default is None.

        Returns
        -------
        None
        """

        title, xlabel, ylabel = self.__get_plot_attributes(title, xlabel, ylabel)
        
        self.__standard_figure_size()
        plt.scatter(self.__X_train, self.__y_train, color = 'lightgray', label='Data Points')
        plt.plot(self.__X_train, self._intercept + self._slope * self.__X_train, color = 'red', label='Fitted Regression Line')

        plt.scatter(self.__X_test, self.__y_pred, color='darkgreen', label='Predicted Data Points')
        plt.scatter(self.__X_test, self.__y_test, color='blue', label='Testing Data Points')

        # Lines connecting predicted points to true points
        for x_pred, y_pred, x_test, y_test in zip(self.__X_test, self.__y_pred, self.__X_test, self.__y_test):
            plt.plot([x_pred, x_test], [y_pred, y_test], color='gray', linestyle='--', linewidth=1.5)
        
        self.__plot_attributes(title, xlabel, ylabel)

    def __get_plot_attributes(self, title : str = None, xlabel : str = None, ylabel : str = None) -> tuple:
        
        """
        Get attributes for plotting.

        Parameters
        ----------
        title : str, optional
            Title of the plot. The default is None.
        xlabel : str, optional
            Label for the x-axis. The default is None.
        ylabel : str, optional
            Label for the y-axis. The default is None.

        Returns
        -------
        tuple
            A tuple containing the title, xlabel, and ylabel for plotting.
        """
        
        if title is None:
            title = 'Training Data'
        
        if xlabel is None and self.__x_train_col_name is None:
            xlabel = 'X-label Data'
        else:
            xlabel = self.__x_train_col_name
            
        if ylabel is None and self.__y_train_col_name is None:
            ylabel = 'Y-label Data'
        else:
            ylabel = self.__y_train_col_name
            
        return (title, xlabel, ylabel)
    
    def __plot_attributes(self, title : str = None, xlabel : str = None, ylabel : str = None) -> None:
        
        """
        Set attributes for plotting.

        Parameters
        ----------
        title : str, optional
            Title of the plot. The default is None.
        xlabel : str, optional
            Label for the x-axis. The default is None.
        ylabel : str, optional
            Label for the y-axis. The default is None.

        Returns
        -------
        None
        """
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()
        
    def __standard_figure_size(self) -> None:
        
        """
        Set a standard figure size for plotting.

        Returns
        -------
        None
        """
        
        plt.figure(figsize = (10, 8))


    ############################################################################################################
    ########################################### Validating Data ################################################
    ############################################################################################################
    def __check_if_valid_data(self, X_data : pd.core.frame.DataFrame | pd.core.series.Series | np.ndarray, y_data : pd.core.frame.DataFrame | pd.core.series.Series | np.ndarray = None, sample_weigths : np.ndarray | list = None) -> None:
        
        """
        Check if the input data is valid.

        Parameters
        ----------
        X_data : pandas.DataFrame, pandas.Series, numpy.ndarray
            Data for the independent variable.
        y_data : pandas.DataFrame, pandas.Series, numpy.ndarray, optional
            Data for the dependent variable. The default is None.
        sample_weights : numpy.ndarray or list, optional
            Sample weights. The default is None.

        Returns
        -------
        None
        """
        
        # Check if data is a valid type
        if not isinstance(X_data, (pd.core.frame.DataFrame, pd.core.series.Series, np.ndarray)):
            raise TypeError("X_Data must be a pandas DataFrame/Series or a numpy array")

        if not isinstance(y_data, (pd.core.frame.DataFrame, pd.core.series.Series, np.ndarray)) and y_data is not None:
            raise TypeError("Y_Data must be a pandas DataFrame/Series or a numpy array")
        
        if isinstance(X_data, (pd.core.frame.DataFrame, pd.core.series.Series, np.ndarray)) and isinstance(y_data, (pd.core.frame.DataFrame, pd.core.series.Series, np.ndarray)) and len(X_data) != len(y_data):
            raise ValueError("X_Data and Y_Data must be the same length")
        
        if not isinstance(sample_weigths, (np.ndarray, list)) and sample_weigths is not None:
            raise TypeError("Sample weights must be a numpy array or a list")

    def __check_X_train(self, X_train : pd.core.frame.DataFrame | pd.core.series.Series | np.ndarray) -> np.ndarray:
        
        """
        Check and convert the training data for the independent variable.

        Parameters
        ----------
        X_train : pandas.DataFrame, pandas.Series, numpy.ndarray
            Training data for the independent variable.

        Returns
        -------
        np.ndarray
            Converted training data.
        """
        
        if isinstance(X_train, pd.core.frame.DataFrame):
            return X_train.to_numpy(), X_train.columns[0]
        elif isinstance(X_train, pd.core.series.Series):
            return X_train.to_numpy(), X_train.name
        else:
            return X_train
    
    def __check_y_train(self, y_train : pd.core.frame.DataFrame | pd.core.series.Series | np.ndarray) -> np.ndarray:
        
        """
        Check and convert the training data for the dependent variable.

        Parameters
        ----------
        y_train : pandas.DataFrame, pandas.Series, numpy.ndarray
            Training data for the dependent variable.

        Returns
        -------
        np.ndarray
            Converted training data.
        """
        
        if isinstance(y_train, pd.core.frame.DataFrame):
            return y_train.to_numpy(), y_train.columns[0]
        elif isinstance(y_train, pd.core.series.Series):
            return y_train.to_numpy(), y_train.name
        else:
            return y_train
        
    def __check_X_test(self, X_test : pd.core.frame.DataFrame | pd.core.series.Series | np.ndarray) -> np.ndarray:
        
        """
        Check and convert the testing data for the independent variable.

        Parameters
        ----------
        X_test : pandas.DataFrame, pandas.Series, numpy.ndarray
            Testing data for the independent variable.

        Returns
        -------
        np.ndarray
            Converted testing data.
        """
        
        if isinstance(X_test, (pd.core.frame.DataFrame, pd.core.series.Series)):
            return X_test.to_numpy()
        else:
            return X_test
        
    def __check_prediction_testing(self, y_pred : pd.core.frame.DataFrame | np.ndarray | pd.core.series.Series, y_test : pd.core.frame.DataFrame | np.ndarray | pd.core.series.Series) -> tuple:
            
        """
        Check and convert predicted and true values for consistency.

        Parameters
        ----------
        y_pred : pandas.DataFrame, numpy.ndarray, pandas.Series
            Predicted values for the dependent variable.
        y_test : pandas.DataFrame, numpy.ndarray, pandas.Series
            True values for the dependent variable.

        Returns
        -------
        tuple
            A tuple containing the converted predicted and true values.
        """    
        
        if not isinstance(y_pred, (pd.core.frame.DataFrame, np.ndarray, pd.core.series.Series)):
            raise TypeError('y_pred must be a numpy array or a pandas series/dataframe')
        
        if not isinstance(y_test, (pd.core.frame.DataFrame, np.ndarray, pd.core.series.Series)):
            raise TypeError('y_test must be a numpy array or a pandas series/dataframe')
        
        if len(y_pred) != len(y_test):
            raise ValueError('y_pred and y_test must be the same length')
        
        if isinstance(y_pred, (pd.core.frame.DataFrame, pd.core.series.Series)):
            y_pred = y_pred.to_numpy()
        
        if isinstance(y_test, (pd.core.frame.DataFrame, pd.core.series.Series)):
            y_test = y_test.to_numpy()
            
        return (y_pred, y_test)
        
    def __check_metric(self, metric : str) -> None:
        
        """
        Check if the specified metric is supported.

        Parameters
        ----------
        metric : str
            The metric to check.

        Returns
        -------
        None
        """
        
        if metric not in ['mean_squared_error', 'R_squared', "MSE", "R2"]:
            raise ValueError('Metric not supported, please choose from mean_squared_error or R_squared or MSE or R2')



if __name__ == "__main__":

    df = pd.read_csv('../datasets/advertising.csv')

    # split the data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(df['TV'], df['Sales'], train_size = 0.85, test_size = 0.15, random_state = 100)
    
    # create an instance of the class
    lr = SimpleLinearRegression()
    
    # fit the model
    lr.fit(X_train, y_train)
    lr.plot_scatter(title = "TV vs Sales")
    
    # print the coefficients
    print("Intercept:", lr._intercept)
    print("Slope:", lr._slope)
    
    # plot the regression line
    lr.plot_regression_line(title = "TV vs Sales")

    # this plot can only be created after the regression line is trained
    lr.plot_residuals(title="TV vs Sales", both = True)

    # predict the values for the testing data
    predictions = lr.predict(X_test)
    print(predictions)

    # calculate the metrics
    MSE = lr.get_metric(predictions, y_test, 'MSE')
    print("MSE:", MSE)

    R_squared = lr.get_metric(predictions, y_test, 'R_squared')
    print("R_squared:", R_squared)


    # this plot can be created only after the predictions are made and a metric is calculated
    lr.plot_testing_and_prediction(title="TV vs Sales")