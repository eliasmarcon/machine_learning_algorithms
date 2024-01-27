import os
import shutil
import tempfile
import time
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
from IPython.display import display, clear_output

import warnings
warnings.filterwarnings('ignore') 

class KMeans():
    
    ############################################################################################################
    ################################### Initialization / Main Functions ########################################
    ############################################################################################################    
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
        
        # Initialize colors for 2d plots (a colormap to get distinct colors for each cluster)
        self.__colors_2d_plots = cm.rainbow(np.linspace(0, 1, self.__k))
        
        
        # Set the random state if not provided
        if random_state is None: 
            random_state = np.random.randint(1, 101)
        
        random.seed(random_state)
        np.random.seed(random_state)
        
    def fit(self, data : pd.DataFrame | np.ndarray, scaling_method : str = None) -> None:
        
        """
        Fit the KMeans model to the data.
        
        Parameters
        ----------
        data : pd.DataFrame | np.ndarray
            Data to be clustered.
        scaling_method : str, optional
            Scaling method to use. The default is None. Options are 'min_max', 'standardization', 'mean_normalization'
        """
        
        # Check if the data is valid
        self.__check_if_valid_data(data)
        
        # Get the number of columns in the data
        self.__n_columns = data.shape[1]
        
        # Set column names for the data (or generate default names if not a DataFrame)
        self.__column_names = data.columns if type(data) == pd.core.frame.DataFrame else ["Column " + str(i) for i in range(1, self.__n_columns + 1)]
        
        # Ensure data is in numpy array format
        self.__data = self.__check_if_dataframe(data)
               
        # Scale the data if a scaling method is specified
        if scaling_method: 
            self.__apply_scaling_method(scaling_method)
    
    def perform(self, show_initial_centroids : bool = False, plot_data : bool = False, gif_path : str = None) -> None:
        
        """
        Perform KMeans clustering and visualize the process.
        
        Parameters
        ----------
        show_initial_centroids : bool, optional
            Whether to show the initial centroids. The default is False.
        plot_data : bool, optional
            Whether to plot the data. The default is False.
        gif_path : str, optional
            Path to save the GIF. The default is None.
        """
        
        # Initialize visualization parameters
        self.__show_initial_centroids = show_initial_centroids
        self.__output_file = gif_path
        self.__plot_array = []
        
        # Validate plot dimensions for 2D or 3D plots
        if plot_data and (self.__n_columns < 1 or self.__n_columns > 3): 
            raise ValueError("Plot dimension must be 2 or 3 (for 2D plots or 3D plots, respectively)")
        
        # Initialize temporary directory for saving plot images
        if self.__output_file is not None: 
            self.__temp_dir = tempfile.mkdtemp()
        
        # Plot initial data for 2D or 3D
        if plot_data and self.__n_columns == 2:
            self.__plot_initial_data()
        
        elif plot_data and self.__n_columns == 3:    
            
            try:
                if self.__check_if_python_script():
                    self.__show_initial_centroids = False
                    plot_data = False
                    raise ValueError("3D plotly plots are not well represented in Python scripts")
            except ValueError as e:
                print(e)
            else:
                self.__plot_initial_data_3d()
                       
        # Get initial centroids
        centroids = self.__apply_cluster_method()
                
        prev_clusters = np.zeros(len(self.__data))
        no_change_count = 0
        
        # Iterative process to update centroids and visualize
        while self.__iteration <= self.__max_iter and no_change_count < self.__patience:
            
            # Assign each data point to the closest centroid
            cluster_array = self.__add_data_point_to_cluster(centroids)
            
            # Check for convergence
            if np.array_equal(prev_clusters, cluster_array): 
                no_change_count += 1
            
            else: 
                no_change_count = 0
            
            prev_clusters = cluster_array.copy()
            
            # Update visualization for 2D or 3D
            if plot_data and self.__n_columns == 2:
                self.__update_plot(centroids, cluster_array)
            
            elif plot_data and self.__n_columns == 3:
                self.__update_plot_3d(centroids, cluster_array)
            
            # Update centroids
            centroids = self.__update_centroids(cluster_array)
            
            self.__iteration += 1
        
        # Display final plot for 2D
        if plot_data and self.__n_columns == 2:
            plt.show()
        
        # Save the final state
        self.__cluster_array = cluster_array
        self.__centroids = centroids

        # Save GIF if requested
        if self.__output_file is not None and (self.__show_initial_centroids or plot_data): 
            self.__create_gif()
            
    def predict(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        
        """
        Predict the clusters for new data.
        
        Parameters
        ----------
        data : pd.DataFrame | np.ndarray
            Data to be clustered.
        """
        
        # Check if the data is valid
        self.__check_if_valid_data(data)
    
        # Check if the number of columns matches the training data
        if data.shape[1] != self.__n_columns: 
            raise ValueError(f"Data must have {self.__n_columns} columns")

        # Make predictions for new data
        return self.__predict_new_data(self.__check_if_dataframe(data))

    def get_cluster_array(self, visualize: bool = False) -> np.ndarray:

        # Visualize the cluster array
        if visualize:
            self.__plot_cluster_array() 
        
        return self.__cluster_array
        
    def get_centroids(self) -> np.ndarray:  

        # Return the centroids
        return self.__centroids

     
    ############################################################################################################
    ############################################# Scaling Functions ############################################
    ############################################################################################################
    def __apply_scaling_method(self, scaling_method: str) -> None:
        
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
        if scaling_method == 'min_max':
        
            # Min-Max scaling: scale to the range [0, 1]
            self.__data = (self.__data - self.__data.min()) / (self.__data.max() - self.__data.min())

        elif scaling_method == 'standardization':
        
            # Standardization: scale to have mean=0 and standard deviation=1
            self.__data = (self.__data - self.__data.mean()) / self.__data.std()

        elif scaling_method == 'mean_normalization':
        
            # Mean normalization: scale to have mean=0 and range [-1, 1]
            self.__data = (self.__data - self.__data.mean()) / (self.__data.max() - self.__data.min())
                
        else:
        
            # Unsupported scaling method, raise an error
            raise ValueError(f"Unsupported scaling method: {scaling_method}, supported methods are: min_max, standardization, mean_normalization")


    ############################################################################################################
    ############################################## Cluster Method ##############################################
    ############################################################################################################
    def __apply_cluster_method(self) -> None:
        
        """
        Apply the specified cluster method to initialize centroids.

        Returns
        -------
        None
        """
        
        if self.__cluster_method == 'kmeans++':
            return self.__k_means_plus_plus()
        
        elif self.__cluster_method == 'random':
            return self.__random_centroids()
        
        else:
            raise ValueError(f"Unsupported cluster method: {self.__cluster_method}, supported cluster methods are: kmeans++, random")

    def __k_means_plus_plus(self) -> np.ndarray:
        
        """
        Initialize centroids using the KMeans++ method.

        Returns
        -------
        np.ndarray
            Array containing the initialized centroids.
        """
        
        # Initialize centroids with one randomly chosen data point
        centroids = self.__data[np.random.choice(len(self.__data), 1, replace=False)]

        # Update and visualize centroids if required
        if self.__show_initial_centroids and self.__n_columns == 2:
            
            self.__update_initial_centroids(centroids)
            self.__save_plot_as_png(f'initial_centroids')
       
        elif self.__show_initial_centroids and self.__n_columns == 3:
            self.__update_initial_centroids_3d(centroids)

        # Iterate to select the remaining centroids
        for k in range(self.__k - 1):
            
            # Initialize a list to store distances of data points from the nearest centroid
            dist = []

            # Compute distance of each data point from the nearest centroid
            for i in range(self.__data.shape[0]):
                
                point = self.__data[i, :]
                d = np.inf

                # Compute the distance of 'point' from each of the previously selected centroids
                for j in range(len(centroids)):
                    temp_dist = self.__get_distance(point, centroids[j])
                    d = min(d, temp_dist)

                dist.append(d)

            # Select the data point with the maximum distance as the next centroid
            centroids = np.append(centroids, [self.__data[np.argmax(np.array(dist)), :]], axis=0)

            # Update and visualize centroids if required
            if self.__show_initial_centroids and self.__n_columns == 2:
                
                self.__update_initial_centroids(centroids)
                self.__save_plot_as_png(f'iteration_centroids_{k}')
           
            elif self.__show_initial_centroids and self.__n_columns == 3:
                self.__update_initial_centroids_3d(centroids)

        return centroids

    def __random_centroids(self) -> np.ndarray:
        
        """
        Initialize centroids using the random method.

        Returns
        -------
        np.ndarray
            Array containing the initialized centroids.
        """
        
        # Randomly choose data points as centroids
        centroids = self.__data[np.random.choice(len(self.__data), self.__k, replace=False)]

        # Update and visualize centroids if required
        if self.__show_initial_centroids and self.__n_columns == 2:
            
            self.__update_initial_centroids(centroids)
            self.__save_plot_as_png(f'initial_centroids')
        
        elif self.__show_initial_centroids and self.__n_columns == 3:
            self.__update_initial_centroids_3d(centroids)

        return centroids
    
    
    ############################################################################################################
    ############################################ Centroid Functions ############################################
    ############################################################################################################
    def __update_centroids(self, clusters: np.ndarray) -> np.ndarray:
        
        """
        Update centroids based on the current assignment of data points to clusters.

        Parameters
        ----------
        clusters : np.ndarray
            Array containing the cluster assignments for each data point.

        Returns
        -------
        np.ndarray
            Array containing the updated centroids.
        """
        
        # Initialize a new array to store the updated centroids
        new_centroids = np.zeros((self.__k, self.__n_columns))
        
        # Iterate over each cluster
        for i in range(self.__k):
            # Calculate the mean of data points belonging to the current cluster
            new_centroids[i] = np.mean(self.__data[clusters == i], axis=0)

        return new_centroids

    def __add_data_point_to_cluster(self, centroids: np.ndarray) -> np.ndarray:
        
        """
        Assign each data point to the closest centroid.

        Parameters
        ----------
        centroids : np.ndarray
            Array containing the current centroids.

        Returns
        -------
        np.ndarray
            Array containing the assigned cluster index for each data point.
        """
        
        # Initialize an array to store distances between each data point and each centroid
        distances = np.zeros((len(self.__data), self.__k))
        
        # Calculate distances for each data point to each centroid
        for i, centroid in enumerate(centroids):
            distances[:, i] = self.__get_distance(self.__data, centroid)

        # Find the index of the centroid with the minimum distance for each data point
        return np.argmin(distances, axis=1)
        
        
    ############################################################################################################
    ############################################ Distance Metrics ##############################################
    ############################################################################################################    
    def __get_distance(self, points: np.ndarray, centroid: np.ndarray) -> None:
        
        """
        Calculate the distance between data points and a centroid based on the specified metric.

        Parameters
        ----------
        points : np.ndarray
            Array containing the data points.
        centroid : np.ndarray
            Array containing the centroid.

        Returns
        -------
        np.ndarray
            Array containing the calculated distances.

        Raises
        ------
        ValueError
            If the specified distance metric is not supported.
        """
        
        new_axis = 0 if len(points.shape) == 1 else 1
        
        if self.__distance_metric == 'euclidean':
            
            # Euclidean distance: square root of the sum of squared differences
            return np.sqrt(np.sum(np.square(points - centroid), axis=new_axis))
        
        elif self.__distance_metric == 'manhattan':
           
            # Manhattan distance: sum of absolute differences
            return np.sum(np.abs(points - centroid), axis=new_axis)
        
        elif self.__distance_metric == 'squared_euclidean':
           
            # Squared Euclidean distance: sum of squared differences
            return np.sum(np.square(points - centroid), axis=new_axis)
        
        elif self.__distance_metric == 'canberra':
            
            # Canberra distance: sum of absolute differences normalized by the sum of absolute values
            return np.sum(np.abs(points - centroid) / (np.abs(points) + np.abs(centroid)), axis=new_axis)
        
        else:
            
            # Unsupported distance metric, raise an error
            raise ValueError(f"Unsupported distance metric: {self.__distance_metric}, supported distance metrics are: euclidean, manhattan, squared_euclidean, canberra")
        
        
    ############################################################################################################
    ############################################# New Predictions ##############################################
    ############################################################################################################    
    def __predict_new_data(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        
        """
        Predict the clusters for new data.

        Parameters
        ----------
        data : pd.DataFrame | np.ndarray
            Data to be clustered.

        Returns
        -------
        np.ndarray
            Array containing cluster assignments for each data point.
        """
        
        # Initialize an array to store distances between data points and centroids
        distances = np.zeros((len(data), self.__k))
        
        # Iterate through centroids and calculate distances
        for i, centroid in enumerate(self.__centroids):
            # Calculate distances using the __get_distance method
            distances[:, i] = self.__get_distance(data.values, centroid)
        
        # Return the index of the minimum distance (cluster assignment)
        return np.argmin(distances, axis=1)
    
    
    
    ########################################################################################################################################################################################################################
    ###################################################################################################### PLOTTING ########################################################################################################
    ########################################################################################################################################################################################################################
    
    
    ############################################################################################################
    ######################################### Plotting Cluster Array ###########################################
    ############################################################################################################
    def __plot_cluster_array(self) -> None:
        
        """
        Plot a bar chart representing the distribution of data points across clusters.

        This method creates a bar chart to visualize the distribution of data points
        across different clusters in the cluster array.

        Returns
        -------
        None
        """
        
        # Create a figure for plotting the cluster array
        fig = plt.figure(figsize=(8, 6))

        # Count occurrences of each unique value in the cluster array
        unique_values, counts = np.unique(self.__cluster_array, return_counts=True)

        # Create a color map for different colors
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_values)))

        # Create a bar chart with different colors for each cluster
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
    ################################## Plotting initial Centroids Functions ####################################
    ############################################################################################################
    def __update_initial_centroids(self, centroids : np.ndarray) -> None:
        
        """
        Update the plot with the newly selected centroids during the KMeans++ initialization.

        This method updates the plot with the newly selected centroids during the KMeans++
        initialization process. It removes the data points that match the centroids to prevent
        overlap and uses distinct markers and colors for visualization.

        Parameters
        ----------
        centroids : np.ndarray
            Array containing the newly selected centroids.

        Returns
        -------
        None
        
        """
        self.__ax.clear()  # Clear the axes

        # Find rows that are equal to the referenced centroids
        mask = np.any(np.all(self.__data == centroids[:, np.newaxis], axis=2), axis=0)

        # Use boolean indexing to exclude centroids based on the mask
        result_array = self.__data[~mask]

        # Scatter plot for data points
        self.__ax.scatter(result_array[:, 0], result_array[:, 1], marker='o', label='Data Points')

        # Plot previously selected centroids for KMeans++
        if self.__cluster_method == "kmeans++":
            
            self.__ax.scatter(centroids[:-1, 0], centroids[:-1, 1], marker='^', s=150, color='orange', label='Previously\nselected\ncentroids')
            self.__ax.scatter(centroids[-1, 0], centroids[-1, 1], marker='*', s=150, color='red', label='Next centroid')

        # Plot next centroids for random initialization
        else:
            self.__ax.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=150, color=self.__colors_2d_plots, label='Next centroid')

        # Set title for the plot
        self.__ax.set_title(f'Select {centroids.shape[0]} th centroid')

        # Call helper methods for additional plot attributes
        self.__plot_attributes()
        
        # Call helper method for overall plot layout
        self.__plot_layout()
        
        
    ############################################################################################################
    ################################## Plotting initial Centroids Functions 3d #################################
    ############################################################################################################
    def __update_initial_centroids_3d(self, centroids : np.ndarray) -> None:
        
        """
        Update the 3D plot with the newly selected centroids during the KMeans++ initialization.

        This method updates the 3D plot with the newly selected centroids during the KMeans++
        initialization process. It removes data points that match the centroids to prevent
        overlap and uses distinct markers and colors for visualization.

        Parameters
        ----------
        centroids : np.ndarray
            Array containing the newly selected centroids.

        Returns
        -------
        None
        """
        
        # Find rows that are equal to the reference centroids
        mask = np.any(np.all(self.__data == centroids[:, np.newaxis], axis=2), axis=0)

        # Use boolean indexing to exclude centroids based on the mask
        result_array = self.__data[~mask]

        # Clear previous traces
        self.__fig.data = []

        # Create a 3D scatter plot for data points
        self.__fig = go.Figure(data=[go.Scatter3d(
            x=result_array[:, 0],
            y=result_array[:, 1],
            z=result_array[:, 2],
            mode='markers',
            marker=dict(size=5, color='blue'),
            name='Data Points'
        )])

        # Plot previously selected centroids for KMeans++
        if self.__cluster_method == "kmeans++":
            
            self.__fig.add_trace(go.Scatter3d(
                x=centroids[:-1, 0],
                y=centroids[:-1, 1],
                z=centroids[:-1, 2],
                mode='markers',
                marker=dict(size=8, symbol='square', color='orange', line=dict(color='black', width=2)),
                name='Previously\nselected\ncentroids'
            ))

            self.__fig.add_trace(go.Scatter3d(
                x=centroids[-1:, 0],
                y=centroids[-1:, 1],
                z=centroids[-1:, 2],
                mode='markers',
                marker=dict(size=8, symbol='diamond', color='red', line=dict(color='black', width=2)),
                name='Next centroid'
            ))

        else:
            
            # Plot next centroids for random initialization
            self.__fig.add_trace(go.Scatter3d(
                x=centroids[:, 0],
                y=centroids[:, 1],
                z=centroids[:, 2],
                mode='markers',
                marker=dict(size=8, symbol='diamond', color=self.__colors_2d_plots, line=dict(color='black', width=2)),
                name='Centroids'
            ))

        self.__fig.update_layout(scene=dict(zaxis=dict(showticklabels=False)), title_text=f'Select {centroids.shape[0]} th centroid')
        self.__plot_attributes_3d()

    def __plot_initial_data(self) -> None:
        
        """
        Plot the initial state of the data in 2D.

        This method creates a 2D scatter plot for the initial state of the data.

        Returns
        -------
        None
        """
        
        self.__fig, self.__ax = plt.subplots(figsize=(10, 8))
        self.__ax.scatter(self.__data[:, 0], self.__data[:, 1], marker='o', label='Data Points')
        self.__ax.set_title('Initial State')
        self.__plot_attributes()

        # Check if running in a Python script
        if self.__check_if_python_script():
            
            plt.tight_layout()
            plt.show(block=False)
            plt.waitforbuttonpress(0)
        
        else:
            self.__plot_layout()

        self.__save_plot_as_png("initial_state")

    def __update_plot(self, centroids : np.ndarray, clusters : np.ndarray) -> None:
        
        """
        Update the plot with the current state during the KMeans clustering iteration in 2D.

        This method updates the plot with the current state during the KMeans clustering
        iteration in 2D. It shows the data points, clusters, and centroids with distinct markers
        and colors for visualization.

        Parameters
        ----------
        centroids : np.ndarray
            Array containing the current centroids.
        clusters : np.ndarray
            Array containing the current cluster assignments.

        Returns
        -------
        None
        """
        
        self.__ax.clear()  # Clear the axes

        for i in range(self.__k):
            
            # Scatter plot for data points in each cluster
            cluster_points = self.__data[clusters == i]
            self.__ax.scatter(cluster_points[:, 0], cluster_points[:, 1], marker='o', label=f'Cluster {i}', color=self.__colors_2d_plots[i])

            # Scatter plot for centroids with the color of the corresponding cluster
            self.__ax.scatter(centroids[i, 0], centroids[i, 1], marker='*', s=200, c=self.__colors_2d_plots[i], edgecolors='black', label=f'Centroid {i}')

        self.__ax.set_title(f'Current State Iteration: {self.__iteration}')
        self.__plot_attributes()
        self.__plot_layout()

        # Save the plot as a PNG image
        self.__save_plot_as_png(f'iteration_{self.__iteration}')

    def __plot_attributes(self) -> None:
        
        """
        Set plot attributes for 2D plots.

        This method sets legend, labels, and other attributes for 2D plots.

        Returns
        -------
        None
        """
        
        self.__ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        self.__ax.set_xlabel(self.__column_names[0])
        self.__ax.set_ylabel(self.__column_names[1])

    def __plot_layout(self) -> None:
        
        """
        Set the plot layout for 2D plots.

        This method sets the layout for 2D plots, including tight layout for Python scripts.

        Returns
        -------
        None
        """
       
        if self.__check_if_python_script():
            
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(1)
        
        else:
            
            display(self.__fig)
            self.__sleep_time()
            clear_output(wait=True)
        
        
    ############################################################################################################
    ####################################### Saving Plots & creating Gif ########################################
    ############################################################################################################
    def __save_plot_as_png(self, step : str) -> None:
        
        """
        Save the current plot as a PNG file.

        This method saves the current plot as a PNG file in a temporary directory.

        Parameters
        ----------
        step : str
            Identifier or step name for the saved plot.

        Returns
        -------
        None
        """
        
        if self.__output_file is not None:
            
            # Create the file path for the PNG file
            png_file = os.path.join(self.__temp_dir, f'{step}.png')

            # Save the current plot as a PNG file
            self.__fig.savefig(png_file)

            # Add the file path to the list for GIF creation
            self.__plot_array.append(png_file)

    def __create_gif(self) -> None:
        
        """
        Create a GIF from the saved PNG files.

        This method generates a GIF from the saved PNG files in the temporary directory.

        Returns
        -------
        None
        """
        
        # Create a list of Image objects from the saved PNG files
        images = [Image.open(png_file) for png_file in self.__plot_array]

        # Generate a GIF from the PNG files with a duration of 500 milliseconds per frame and looping
        imageio.mimsave(self.__output_file + '.gif', images, duration=500, loop=0)

        # Clean up: Remove the temporary directory and its contents
        shutil.rmtree(self.__temp_dir)


    ############################################################################################################
    ########################################## Plotting Functions 3d ###########################################
    ############################################################################################################ 
    def __plot_initial_data_3d(self) -> None:
        
        """
        Plot the initial state of 3D data.

        This method creates a 3D scatter plot for the initial state of the data.

        Returns
        -------
        None
        """
        
        # Create a subplot with a 3D scatter plot for data points
        self.__fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

        scatter = go.Scatter3d(
            x=self.__data[:, 0],
            y=self.__data[:, 1],
            z=self.__data[:, 2],
            mode='markers',
            marker=dict(size=5, color='blue'),
            name='Data Points'
        )

        # Add the 3D scatter plot to the subplot
        self.__fig.add_trace(scatter)

        # Customize the layout for the 3D plot
        self.__fig.update_layout(scene=dict(zaxis=dict(showticklabels=False)), title_text='Initial State')

        # Add attributes and display the plot
        self.__plot_attributes_3d()

    def __update_plot_3d(self, centroids: np.ndarray, clusters: np.ndarray) -> None:
        
        """
        Update the 3D plot with new centroids and clusters.

        This method updates the 3D scatter plot with new centroids and clusters for the current iteration.

        Parameters
        ----------
        centroids : np.ndarray
            Array containing the coordinates of centroids.
        clusters : np.ndarray
            Array containing the cluster assignments for each data point.

        Returns
        -------
        None
        """
        
        # Clear previous traces in the 3D plot
        self.__fig.data = []

        # Define colors for clusters
        colors = px.colors.qualitative.G10

        for i in range(self.__k):
            
            # Get data points for the current cluster
            cluster_points = self.__data[clusters == i]

            # Create a scatter plot for the cluster
            scatter_cluster = go.Scatter3d(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                z=cluster_points[:, 2],
                mode='markers',
                marker=dict(size=5, color=colors[i]),
                name=f'Cluster {i}'
            )

            # Add the cluster scatter plot to the 3D plot
            self.__fig.add_trace(scatter_cluster)

            # Create a scatter plot for the centroid
            scatter_centroid = go.Scatter3d(
                x=[centroids[i, 0]],
                y=[centroids[i, 1]],
                z=[centroids[i, 2]],
                mode='markers',
                marker=dict(size=8, color=colors[i], symbol='diamond', line=dict(color='black', width=2)),
                name=f'Centroid {i}'
            )

            # Add the centroid scatter plot to the 3D plot
            self.__fig.add_trace(scatter_centroid)

        # Update layout for the 3D plot
        self.__fig.update_layout(scene=dict(zaxis=dict(showticklabels=False)), title_text=f'Current State Iteration: {self.__iteration}')

        # Add attributes and display the updated 3D plot
        self.__plot_attributes_3d()

    def __plot_attributes_3d(self) -> None:
        
        """
        Customize layout attributes for the 3D plot.

        This method customizes layout attributes for the 3D plot, including axis titles.

        Returns
        -------
        None
        """
        
        # Define layout settings for the 3D plot
        layout = go.Layout(
            scene=dict(
                xaxis=dict(title=self.__column_names[0]),
                yaxis=dict(title=self.__column_names[1]),
                zaxis=dict(title=self.__column_names[2])
            ),
            width=1200,
            height=1000
        )

        # Update the layout for the 3D plot
        self.__fig.update_layout(layout)

        # Show the 3D plot, wait for a short time, and then clear the output
        self.__fig.show()
        self.__sleep_time()
        clear_output(wait=True)
        
        
    ############################################################################################################
    ############################################# Helper Functions #############################################
    ############################################################################################################
    def __check_if_valid_data(self, data: pd.DataFrame | np.ndarray) -> None:
        
        """
        Check if the input data is a valid type.

        Parameters
        ----------
        data : pd.DataFrame | np.ndarray
            The input data to be checked.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the data is not a pandas DataFrame or a numpy array.
        """
        
        # Check if data is a valid type
        if not isinstance(data, (pd.core.frame.DataFrame, np.ndarray)):
            raise ValueError("Data must be a pandas DataFrame or a numpy array")

    def __check_if_dataframe(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        
        """
        Convert data to a numpy array if it is a DataFrame.

        Parameters
        ----------
        data : pd.DataFrame | np.ndarray
            The input data to be converted.

        Returns
        -------
        np.ndarray
            The data as a numpy array.
        """
        
        # Convert data to numpy array if it is a DataFrame
        if type(data) == pd.core.frame.DataFrame:
            return data.values
        
        else:
            return data

    def __sleep_time(self) -> None:
        
        """
        Sleep for a short period of time.

        This method introduces a pause in the program's execution for one second.

        Returns
        -------
        None
        """
        
        time.sleep(1)

    def __check_if_python_script(self) -> bool:
        
        """
        Check if the code is being run as a standalone Python script.

        Returns
        -------
        bool
            True if running as a python script, False otherwise.
        """
        
        return "__name__" in globals() and __name__ == "__main__"


############################################################################################################
############################################## Example Usage ###############################################
############################################################################################################
if __name__ == "__main__":
    
    df_customers = pd.read_csv('../datasets/Mall_Customers.csv') 
    df_customers.drop(['CustomerID', 'Gender'], axis=1, inplace=True)

    kmeans = KMeans(k = 4, max_iter = 10, cluster_method = 'kmeans++', distance_metric='manhattan', random_state = 100)
    kmeans.fit(df_customers[['Annual Income (k$)', 'Spending Score (1-100)']], scaling_method='standardization') #, 'Age'
    kmeans.perform(show_initial_centroids=True, plot_data=True)

    print("Cluster Array\n", kmeans.get_cluster_array(visualize=True), "\n")
    print("Cluster Centroids\n", kmeans.get_centroids())