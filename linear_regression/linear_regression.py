import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression


def separator():

    print("\n")


def read_present_data():

    df_sales = pd.read_csv('advertising.csv') # https://www.kaggle.com/datasets/thorgodofthunder/tvradionewspaperadvertising/data

    print("=" * 30, "Data Overview", "=" * 30, "\n")

    print(df_sales.head(), "\n")
    print(df_sales.info(), "\n")
    print(df_sales.describe(), "\n")

    return df_sales


def data_analysis(df_sales):

    print("=" * 30, "Null Values overview", "=" * 30, "\n")
    null_values = df_sales.isnull().sum()*100 / df_sales.shape[0]
    print(null_values)
    
    # Outlier Analysis
    fig, axs = plt.subplots(3, figsize = (8, 5))
    sns.boxplot(df_sales['TV'], ax = axs[0], orient = "h")
    sns.boxplot(df_sales['Newspaper'], ax = axs[1], orient = "h")
    sns.boxplot(df_sales['Radio'], ax = axs[2], orient = "h")
    plt.tight_layout()
    plt.show()


    sns.pairplot(df_sales, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', height=4, aspect = 1, kind='scatter')
    plt.show()


def create_split(df, column):

    X = df[column]
    y = df['Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.85, test_size = 0.15, random_state = 100)

    return X_train, X_test, y_train, y_test


def model_building(df, target):

    print("=" * 30, "OLS Regression Results", "=" * 30, "\n")

    data = sm.add_constant(df.drop(target, axis=1))
    target = df[target]
    
    ols = sm.OLS(target, data).fit()
    
    features = ['intercept'] + list(df.drop(target.name, axis=1).columns)

    print("Linear Regression Parameters with SM OLS")
    print(ols.summary(xname=features), "\n")

    ########################################################################################################################

    print("=" * 30, "Linear Regression Results", "=" * 30, "\n")

    linear_regression_sklearn = LinearRegression()
    linear_regression_sklearn.fit(data, target)

    print("Linear Regression Parameters with Sklearn")
    print(f"Intercept: {linear_regression_sklearn.intercept_:.4f}")
    print("Coefficients:")
    [print(f"  Coefficient {features[i + 1]}: {coef:.4f}") for i, coef in enumerate(linear_regression_sklearn.coef_[1:])]
    separator()

    ########################################################################################################################
    
    print("=" * 30, "P-Value Results", "=" * 30, "\n")
    
    # Find the feature with the smallest p-value
    print("All p-values:\n")
    print(ols.pvalues[1:], "\n")
    
    best_feature_name = ols.pvalues[1:].idxmin()
    print("Best Feature with smallest p-value:", best_feature_name)
    print(f"Smallest p-value: {ols.pvalues[1:].min()} \n")


def get_models(df, column):

    X_train, X_test, y_train, y_test = create_split(df, column)
    X_train_sm = sm.add_constant(X_train)

    features = ['intercept', column]

    lr = sm.OLS(y_train, X_train_sm).fit()
    lr_params = lr.params.set_axis(features)

    return lr, lr_params, X_train_sm, X_train, y_train, X_test, y_test


def model_building_plots(list_lr_params, list_X_train, list_y_train, col_names):

    fig, axes = plt.subplots(3, figsize = (10, 8))

    for i, ax in enumerate(axes):

        ax.scatter(list_X_train[i], list_y_train[i])
        ax.plot(list_X_train[i], list_lr_params[i][0] + list_lr_params[i][1] * list_X_train[i], 'r')
        ax.set_title('Linear Regression with training set for ' + col_names[i])
        ax.set_xlabel(col_names[i])
        ax.set_ylabel('Sales')

    plt.tight_layout()
    plt.show()


def model_evaluation(list_lr, list_X_train_sm, list_X_train, list_y_train, col_names):

    fig, axes = plt.subplots(3, 2, figsize=(10, 8))

    for i, ax in enumerate(axes):
        
        y_train_pred = list_lr[i].predict(list_X_train_sm[i])
        res = (list_y_train[i] - y_train_pred)

        # Plot the histogram of error terms
        sns.distplot(res, bins=15, ax=ax[0])
        ax[0].set_title('Residual Terms for ' + col_names[i])
        ax[0].set_xlabel('Residuals')

        # Scatter plot of list_X_train against residuals
        ax[1].scatter(list_X_train[i], res)
        ax[1].set_title('Residuals vs. X_train for ' + col_names[i])
        ax[1].set_xlabel(col_names[i])
        ax[1].set_ylabel('Residuals')

    plt.tight_layout()
    plt.show()


def self_mean_squared_error(y_test, y_pred):

    return ( 1 / len(y_test) ) * ( sum( (y_test - y_pred)**2 ) )


def self_r_squared(y_test, y_pred):

    return 1 - ( sum( (y_test - y_pred)**2 ) / sum( (y_test - y_test.mean())**2 ) )


def model_testing(list_lr, list_lr_params, list_X_test, list_y_test, col_names):

    df = pd.DataFrame()

    for i in range(len(list_lr)):
    
        # Add a constant to X_test
        X_test_sm = sm.add_constant(list_X_test[i])

        # Predict the y values corresponding to X_test_sm
        y_pred = list_lr[i].predict(X_test_sm)

        mse = mean_squared_error(list_y_test[i], y_pred)
        r_squared = r2_score(list_y_test[i], y_pred)
        
        self_mse = self_mean_squared_error(list_y_test[i], y_pred)
        self_r_sqrt = self_r_squared(list_y_test[i], y_pred)

        # Define the new row data as a dictionary
        new_row = pd.DataFrame({'Type': [col_names[i]], 
                                'Sklearn_MSE': [mse],
                                'Self_MSE': [self_mse],
                                'Sklearn_R_Squared': [r_squared],
                                'Self_R_Squared': [self_r_sqrt]})

        # Concatenate the original DataFrame and the new row DataFrame
        df = pd.concat([df, new_row], ignore_index=True)

    df.set_index("Type", inplace=True)
    df.index.name = None
    
    print("=" * 30, "Metric Evaluation", "=" * 30, "\n")
    print(df)

    #######################################################################################################

    fig, axes = plt.subplots(3, figsize = (8, 8))

    for i, ax in enumerate(axes):

        ax.scatter(list_X_test[i], list_y_test[i])
        ax.plot(list_X_test[i], list_lr_params[i][0] + list_lr_params[i][1] * list_X_test[i], 'r')
        ax.set_title('Linear Regression with test set for ' + col_names[i])
        ax.set_xlabel(col_names[i])
        ax.set_ylabel('Sales')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    #############################################################################################################################################################################
    ############################################################### Data Loading and Visualizing ################################################################################
    #############################################################################################################################################################################

    df_sales = read_present_data()
    col_names = df_sales.columns
    separator()
    
    data_analysis(df_sales)
    separator()

    #############################################################################################################################################################################
    ###################################################################### Model Building #######################################################################################
    #############################################################################################################################################################################

    model_building(df_sales, 'Sales')

    lr_tv, lr_params_tv, X_train_sm_tv, X_train_tv, y_train_tv, X_test_tv, y_test_tv = get_models(df_sales, col_names[0])

    lr_radio, lr_params_radio, X_train_sm_radio, X_train_radio, y_train_radio, X_test_radio, y_test_radio = get_models(df_sales, col_names[1])

    lr_newspaper, lr_params_newspaper, X_train_sm_newspaper, X_train_newspaper, y_train_newspaper, X_test_newspaper, y_test_newspaper = get_models(df_sales, col_names[2])

    model_building_plots([lr_params_tv, lr_params_radio, lr_params_newspaper],
                         [X_train_tv, X_train_radio, X_train_newspaper], 
                         [y_train_tv, y_train_radio, y_train_newspaper], 
                         col_names[0:3])


    #############################################################################################################################################################################
    ##################################################################### Model Evaluation ######################################################################################
    #############################################################################################################################################################################

    model_evaluation([lr_tv, lr_radio, lr_newspaper],
                     [X_train_sm_tv, X_train_sm_radio, X_train_sm_newspaper], 
                     [X_train_tv, X_train_radio, X_train_newspaper],
                     [y_train_tv, y_train_radio, y_train_newspaper], 
                     col_names[0:3])

    #############################################################################################################################################################################
    ###################################################################### Model Testing ########################################################################################
    #############################################################################################################################################################################

    model_testing([lr_tv, lr_radio, lr_newspaper],
                  [lr_params_tv, lr_params_radio, lr_params_newspaper], 
                  [X_test_tv, X_test_radio, X_test_newspaper],
                  [y_test_tv, y_test_radio, y_test_newspaper], 
                  col_names[0:3])
    
    separator()