import os
import shutil
import tempfile
import imageio
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import plotly.graph_objs as go
import plotly.express as px

from plotly.subplots import make_subplots
from PIL import Image

import warnings
warnings.filterwarnings('ignore') 

class KMeans():
    
    random.seed(10)
    np.random.seed(10)
    
    def __init__(self, k : int = 2, max_iter : int = 100, patience : int = 10, cluster_method : str = 'kmeans++', distance_metric : str = 'euclidean', random_state : int = None):
        
        """
        Parameters
        ----------
        k : int, optional
            Number of clusters. The default is 2.
        max_iter : int, optional
            Maximum number of iterations. The default is 100.
        patience : int, optional
            Number of iterations to wait for convergence. The default is 10.
        cluster_method : str, optional
            Method to initialize the centroids. The default is 'kmeans++'.
        distance_metric : str, optional
            Distance metric to use. The default is 'euclidean'.
        random_state : int, optional
            Random state to use. The default is None.
        """
        
        self.__k = k
        self.__iteration = 1
        self.__max_iter = max_iter
        self.__cluster_method = cluster_method
        self.__patience = patience
        self.__distance_metric = distance_metric
        
        # Set the random state if not provided
        if random_state is None: random_state = np.random.randint(1, 101)
        
        random.seed(random_state)
        np.random.seed(random_state)
        
    def fit(self, data : pd.DataFrame, col_names : list = None, scaling_method : str = None) -> None:
        
        """
        Parameters
        ----------
        data : pd.DataFrame
            Data to be clustered.
        col_names : list, optional
            List of column names to use. The default is None.
        scaling_method : str, optional
            Scaling method to use. The default is None. Options are 'min_max', 'standardization', 'mean_normalization'
        """
        
        if type(data) != pd.core.frame.DataFrame: raise ValueError("Data must be a pandas DataFrame")
        
        if col_names:
        
            self.__n_columns = len(col_names)
            self.__data = data[col_names]
            
        else:
            
            self.__n_columns = len(data.columns)
            self.__data = data
               
        # Scale the data
        if scaling_method: 
            self.__apply_scaling_method(scaling_method)
    
    
    def perform(self, show_initial_centroids : bool = False, plot_data : bool = False, gif_path : str = None) -> None:
        
        """
        Parameters
        ----------
        show_initial_centroids : bool, optional
            Whether to show the initial centroids. The default is False.
        plot_data : bool, optional
            Whether to plot the data. The default is False.
        gif_path : str, optional
            Path to save the GIF. The default is None.
        """
        
        self.__show_initial_centroids = show_initial_centroids
        self.__output_file = gif_path
        self.__plot_array = []
        
        if plot_data and self.__n_columns < 1 or plot_data and self.__n_columns > 3: raise ValueError("Plot dimension must be 2 or 3 (for 2d plots or 3d plots respectively)")
        
        if self.__output_file is not None: self.__temp_dir = tempfile.mkdtemp()
        
        if plot_data and self.__n_columns == 2:
        
            # Plot initial data
            self.__plot_initial_data()
            
        elif plot_data and self.__n_columns == 3:
            
            self.__plot_initial_data_3d()
                       
        # Get initial centroids
        centroids = self.__apply_cluster_method()
        
        prev_clusters = np.zeros(len(self.__data))
        no_change_count = 0
        
        while self.__iteration <= self.__max_iter and no_change_count < self.__patience:
            
            # Assign each data point to the closest centroid
            cluster_array = self.__add_data_point_to_cluster(centroids)
            
            # Check for convergence
            if np.array_equal(prev_clusters, cluster_array):
            
                no_change_count += 1
            
            else:
            
                no_change_count = 0
            
            prev_clusters = cluster_array.copy()
            
            if plot_data and self.__n_columns == 2:
                # Plot current state
                self.__update_plot(centroids, cluster_array)
                
            elif plot_data and self.__n_columns == 3:
                # Plot current state
                self.__update_plot_3d(centroids, cluster_array)
            
            # Update centroids
            centroids = self.__update_centroids(cluster_array)
            
            self.__iteration += 1
        
        if plot_data and self.__n_columns == 2:
            
            plt.show()
            
        elif plot_data and self.__n_columns == 3:
            
            self.create_gif()
        
        # Save the final state
        self.__cluster_array = cluster_array
        self.__centroids = centroids

        if self.__output_file is not None:
            
            # Create GIF
            self.__create_gif()
            

    def predict(self, data : pd.DataFrame) -> np.ndarray:
        
        """
        Parameters
        ----------
        data : pd.DataFrame
            Data to be clustered.
        """
        
        if type(data) != pd.core.frame.DataFrame: raise ValueError("Data must be a pandas DataFrame")
        elif len(data.columns) != self.__n_columns: raise ValueError(f"Data must have {self.__n_columns} columns")
        
        return self.__predict_new_data(data)

    def get_cluster_array(self, visualize : bool = False) -> np.ndarray:
        
        """
        Summary:
        --------
        """
        if visualize:
            
            self.__plot_cluster_array() 
        
        return self.__cluster_array
        

    def get_centroids(self) -> np.ndarray:  
            
        """
        Summary:
        --------
        """
        
        return self.__centroids


    def __plot_cluster_array(self) -> None:
        
        """
        Summary:
        --------
        """
        # Count occurrences of each unique value
        unique_values, counts = np.unique(self.__cluster_array, return_counts = True)

        # Create a color map for different colors
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_values)))

        # Create a bar chart with different colors for each bar
        bars = plt.bar(unique_values, counts, color=colors)

        # Add count values on top of the bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count),
                    ha='center', va='bottom')

        # Customize the plot
        plt.title('Cluster Array')
        plt.xlabel('Cluster')
        plt.xticks(unique_values)
        plt.ylabel('Count')

        # Show the plot
        plt.show()
    
    ############################################################################################################
    ############################################# Scaling Functions ############################################
    ############################################################################################################
    def __apply_scaling_method(self, scaling_method : str) -> None:
        
        """
        Parameters
        ----------
        scaling_method : str
            Scaling method to use. Options are 'min_max', 'standardization', 'mean_normalization'
        """
        if scaling_method == 'min_max':
            
            self.__data = (self.__data - self.__data.min()) / (self.__data.max() - self.__data.min())

        elif scaling_method == 'standardization':
                
            self.__data = (self.__data - self.__data.mean()) / self.__data.std()

        elif scaling_method == 'mean_normalization':
            
            self.__data = (self.__data - self.__data.mean()) / (self.__data.max() - self.__data.min())
            
        else:
                
            raise ValueError(f"Unsupported scaling method: {scaling_method}, supported methods are: min_max, standardization, mean_normalization")

    ############################################################################################################
    ############################################## Cluster Method ##############################################
    ############################################################################################################
    def __apply_cluster_method(self) -> None:
        
        """
        Summary:
        --------
        
        """
        
        if self.__cluster_method == 'kmeans++':

            return self.__k_means_plus_plus()
        
        elif self.__cluster_method == 'random':
            
            return self.__random_centroids()
        
        else:
                
            raise ValueError(f"Unsupported cluster method: {self.__cluster_method}, supported cluster methods are: kmeans++, random")
        
    def __k_means_plus_plus(self) -> np.ndarray:
        
        """
        Summary:
        --------
        """
        
        centroids = self.__data.iloc[np.random.choice(len(self.__data), 1, replace=False)].values
        
        if self.__show_initial_centroids:
            self.__update_initial_centroids(centroids)
            self.__save_plot_as_png(f'initial_centroids')
            
        
        for k in range(self.__k - 1):
            
            ## initialize a list to store distances of data points from nearest centroid
            dist = []

            for i in range(self.__data.shape[0]):
                
                point = self.__data.iloc[i, :].values
                d = np.inf
                
                ## compute distance of 'point' from each of the previously selected centroid and store the minimum distance
                for j in range(len(centroids)):
                        
                    temp_dist = self.__get_distance(point, centroids[j])
                    d = min(d, temp_dist)
                
                dist.append(d)
                
            ## select data point with maximum distance as our next centroid
            centroids = np.append(centroids, [self.__data.iloc[np.argmax(np.array(dist)), :]], axis = 0)

            if self.__show_initial_centroids:
                self.__update_initial_centroids(centroids)
                self.__save_plot_as_png(f'iteration_centroids_{k}')
                
        return centroids
      
    def __random_centroids(self) -> np.ndarray:
        
        """
        Summary:
        --------
        """
        
        centroids = self.__data.iloc[np.random.choice(len(self.__data), self.__k, replace=False)].values
        
        if self.__show_initial_centroids:
            self.__update_initial_centroids(centroids)
            self.__save_plot_as_png(f'initial_centroids')

        return centroids
    
    ############################################################################################################
    ############################################ Centroid Functions ############################################
    ############################################################################################################
    def __update_centroids(self, clusters : np.ndarray) -> np.ndarray:
        
        """
        Summary:
        --------
        """
        
        new_centroids = np.zeros((self.__k, self.__n_columns))
        
        for i in range(self.__k):
            
            new_centroids[i] = np.mean(self.__data[clusters == i], axis = 0)
        
        return new_centroids
    
    def __add_data_point_to_cluster(self, centroids : np.ndarray) -> np.ndarray:
        
        """
        Summary:
        --------
        """
        
        distances = np.zeros((len(self.__data), self.__k))
        
        for i, centroid in enumerate(centroids):

            distances[:, i] = self.__get_distance(self.__data.values, centroid)

        return np.argmin(distances, axis=1)
        
    ############################################################################################################
    ############################################ Distance Metrics ##############################################
    ############################################################################################################    
    def __get_distance(self, points : np.ndarray, centroid : np.ndarray) -> None:
          
        """
        Summary:
        --------
        """
              
        if self.__distance_metric == 'euclidean':
            
            return np.sqrt(np.sum(np.square(points - centroid), axis = 0 if len(points.shape) == 1 else 1)) 

        elif self.__distance_metric == 'manhattan':
            
            # Compute the Manhattan distance
            return np.sum(np.abs(points - centroid), axis=1)
        
        elif self.__distance_metric == 'squared_euclidean':
            
            return np.sum(np.square(points - centroid), axis = 1)
            
        elif self.__distance_metric == 'canberra':
            
            return np.sum(np.abs(points - centroid) / (np.abs(points) + np.abs(centroid)), axis=1)
            
        else:
            
            raise ValueError(f"Unsupported distance metric: {self.__distance_metric}")
        
    ############################################################################################################
    ############################################# New Predictions ##############################################
    ############################################################################################################    
    def __predict_new_data(self, data : pd.DataFrame) -> np.ndarray:
        
        """
        Summary:
        --------
        """
        
        distances = np.zeros((len(data), self.__k))
        
        for i, centroid in enumerate(self.__centroids):

            distances[:, i] = self.__get_distance(data.values, centroid)
        
        return np.argmin(distances, axis=1)
    
    ############################################################################################################
    ################################## Plotting initial Centroids Functions ####################################
    ############################################################################################################
    def __update_initial_centroids(self, centroids : np.ndarray) -> None:
        
        """
        Summary:
        --------
        """
        
        self.ax.clear()  # Clear the axes

        # Function to check if a row is equal to the centroid
        def is_equal_to_centroids(row):
            return any(np.all(row.values == centroid) for centroid in centroids)

        # Apply the function to identify rows equal to the centroid
        mask = self.__data.apply(is_equal_to_centroids, axis=1)
        # Filter the DataFrame to exclude rows equal to the centroid
        result_df = self.__data[~mask]

        self.ax.scatter(result_df.iloc[:, 0], result_df.iloc[:, 1], marker='o', label='Data Points')
        
        if self.__cluster_method == "kmeans++":
            
            self.ax.scatter(centroids[:-1, 0], centroids[:-1, 1], marker='^', s=150, color = 'orange', label = 'Previously selected\n centroids')
            self.ax.scatter(centroids[-1, 0], centroids[-1, 1], marker='*', s=150, color = 'red', label = 'Next centroid')
        
        else:
            self.ax.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=150, color = 'orange', label = 'Next centroid')
        
        self.ax.set_title('Select %d th centroid'%(centroids.shape[0]))
                
        self.__plot_attributes()
        self.__plot_layout()
    
    
    ############################################################################################################
    ########################################## Plotting 2D Functions ###########################################
    ############################################################################################################
    def __plot_initial_data(self) -> None:
        
        """
        Summary:
        --------
        """
        
        self.__fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.scatter(self.__data.iloc[:, 0], self.__data.iloc[:, 1], marker='o', label='Data Points')
        self.ax.set_title('Initial State')
        self.__plot_attributes()
        
        self.__save_plot_as_png("initial_state")
        
        plt.tight_layout()
        plt.show(block=False)  # Display the plot without blocking
        plt.waitforbuttonpress(0)

    def __update_plot(self, centroids : np.ndarray, clusters : np.ndarray) -> None:
        
        """
        Summary:
        --------
        """
        
        self.ax.clear()  # Clear the axes
        
        # Use a colormap to get distinct colors for each cluster
        colors = cm.rainbow(np.linspace(0, 1, self.__k))

        for i in range(self.__k):
            
            cluster_points = self.__data[clusters == i]
            
            self.ax.scatter(cluster_points.iloc[:, 0], cluster_points.iloc[:, 1], marker='o', label=f'Cluster {i}', color=colors[i])

            # Scatter plot for centroids with the color of the corresponding cluster
            self.ax.scatter(centroids[i, 0], centroids[i, 1], marker='*', s=200, c=colors[i], edgecolors='black', label=f'Centroid {i}')

        
        self.ax.set_title(f'Current State Iteration: {self.__iteration}')
        self.__plot_attributes()
        
        self.__save_plot_as_png(f'iteration_{self.__iteration}')
        
        plt.tight_layout()
        plt.show(block=False)  # Display the plot without blocking
        plt.pause(1)

        
    def __plot_attributes(self) -> None:
    
        """
        Summary:
        --------
        """
    
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        self.ax.set_xlabel(self.__data.columns[0])
        self.ax.set_ylabel(self.__data.columns[1])
        
    def __plot_layout(self) -> None:
        
        """
        Summary:
        --------
        """
        
        plt.tight_layout()
        plt.show(block=False)  # Display the plot without blocking
        plt.pause(1)
        
    def __save_plot_as_png(self, step : str) -> None:
           
        """
        Summary:
        --------
        """   
             
        # Save the plot as a PNG file
        if self.__output_file is not None:
            
            png_file = os.path.join(self.__temp_dir, f'{step}.png')
            
            self.__fig.savefig(png_file)
            self.__plot_array.append(png_file)
            
    def __create_gif(self) -> None:
        
        """
        Summary:
        --------
        """
            
        images = [Image.open(png_file) for png_file in self.__plot_array]
            
        # Generate a GIF from the PNG files
        imageio.mimsave(self.__output_file + '.gif', images, duration=500, loop = 0)

        # Clean up: Remove the temporary directory and its contents
        shutil.rmtree(self.__temp_dir)

    ############################################################################################################
    ########################################## Plotting Functions 3d ###########################################
    ############################################################################################################ 
    def __plot_initial_data_3d(self) -> None:
        
        """
        Summary:
        --------
        """
        
        self.__fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
        scatter = go.Scatter3d(
            x=self.__data.iloc[:, 0],
            y=self.__data.iloc[:, 1],
            z=self.__data.iloc[:, 2],
            mode='markers',
            marker=dict(size=5, color='blue'),
            name='Data Points'
        )
        self.__fig.add_trace(scatter)
        self.__fig.update_layout(scene=dict(zaxis=dict(showticklabels=False)), title_text='Initial State')
        self.___plot_attributes_3d()


    def __update_plot_3d(self, centroids : np.ndarray, clusters : np.ndarray) -> None:
        
        """
        Summary:
        --------
        """
        
        self.__fig.data = []  # Clear previous traces
        colors = px.colors.qualitative.G10

        for i in range(self.__k):
            cluster_points = self.__data[clusters == i]
            scatter_cluster = go.Scatter3d(
                x=cluster_points.iloc[:, 0],
                y=cluster_points.iloc[:, 1],
                z=cluster_points.iloc[:, 2],
                mode='markers',
                marker=dict(size=5, color=colors[i]),
                name=f'Cluster {i}'
            )
            self.__fig.add_trace(scatter_cluster)

            scatter_centroid = go.Scatter3d(
                x=[centroids[i, 0]],
                y=[centroids[i, 1]],
                z=[centroids[i, 2]],
                mode='markers',
                marker=dict(size=8, color=colors[i], symbol='diamond', line = dict(color='black', width = 2)),
                name=f'Centroid {i}'
            )
            self.__fig.add_trace(scatter_centroid)

        self.__fig.update_layout(scene=dict(zaxis=dict(showticklabels=False)), title_text=f'Current State Iteration: {self.__iteration}')
        self.___plot_attributes_3d()

    def ___plot_attributes_3d(self) -> None:
        
        """
        Summary:
        --------
        """
        
        layout = go.Layout(
                            scene = dict(
                                    xaxis = dict(title  = self.__data.columns[0]),
                                    yaxis = dict(title  = self.__data.columns[1]),
                                    zaxis = dict(title  = self.__data.columns[2])
                                )
                          )
        
        self.__fig.update_layout(layout)



if __name__ == "__main__":
    
    def read_present_data(synthetic_data = False):

        df_customers = pd.read_csv('../datasets/Mall_Customers.csv') 
        df_customers.drop(['CustomerID', 'Gender'], axis=1, inplace=True)
        
        # Additional synthetic columns
        df_customers['Number of Purchases'] = np.random.randint(1, 20, size=len(df_customers))
        df_customers['Average Purchase Amount'] = np.round(np.random.uniform(10, 1000, size=len(df_customers)), 2)
        
        if synthetic_data:
            synthetic_data_size = 800
            # Additional synthetic rows
            new_rows = {
                'Age': np.random.randint(df_customers['Age'].min(), df_customers['Age'].max(), size=synthetic_data_size),
                'Annual Income (k$)': np.random.randint(df_customers['Annual Income (k$)'].min(), df_customers['Annual Income (k$)'].max(), size=synthetic_data_size),
                'Spending Score (1-100)': np.random.randint(df_customers['Spending Score (1-100)'].min(), df_customers['Spending Score (1-100)'].max(), size=synthetic_data_size),
                'Number of Purchases': np.random.randint(df_customers['Number of Purchases'].min(), df_customers['Number of Purchases'].max(), size=synthetic_data_size),
                'Average Purchase Amount': np.random.uniform(df_customers['Average Purchase Amount'].min(), df_customers['Average Purchase Amount'].max(), size=synthetic_data_size),
            }
            # Concatenate the new data to the existing DataFrame
            df_customers = pd.concat([df_customers, pd.DataFrame(new_rows)], ignore_index=True)

        return df_customers
    
    df_customers = read_present_data()

    # kmeans = KMeans(k = 4, max_iter = 5, cluster_method = 'random', distance_metric='euclidean', random_state = 42)
    # kmeans.fit(df_customers[['Annual Income (k$)', 'Spending Score (1-100)']], scaling_method='standardization')#, 'Average Purchase Amount', 'Spending Score (1-100)'])

    # kmeans.perform(show_initial_centroids=False, plot_data=False)
    # kmeans.perform(col_names=['Annual Income (k$)', 'Spending Score (1-100)', 'Number of Purchases', 'Average Purchase Amount'], plot_dimension=2)

    kmeans = KMeans(k = 4, max_iter = 10, cluster_method = 'kmeans++', distance_metric='euclidean', random_state = 42)
    kmeans.fit(df_customers, scaling_method='standardization')#, 'Average Purchase Amount', 'Spending Score (1-100)'])
    kmeans.perform(show_initial_centroids=False, plot_data=False)
