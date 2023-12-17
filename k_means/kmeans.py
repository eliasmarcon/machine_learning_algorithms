import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt 
import matplotlib.cm as cm

import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

from PIL import Image
from io import BytesIO

import warnings
warnings.filterwarnings('ignore') 

class KMeans():
    
    random.seed(10)
    np.random.seed(10)
    
    def __init__(self, k = 2, max_iter = 100, patience = 10, method = 'kmeans++', distance_metric = 'euclidean', show_initial_centroids = False, random_state = None):
        
        self.k = k
        self.iteration = 1
        self.max_iter = max_iter
        self.method = method
        self.patience = patience
        self.distance_metric = distance_metric
        self.__show_initial_centroids = show_initial_centroids
        
        random.seed(random_state)
        np.random.seed(random_state)
        
    def fit(self, data, col_names = None, scaling_method = 'standardization', plot_data = False):
        
        if col_names is None:
        
            self.n_columns = 2
            
            # Get two random column names without replacement
            random_columns = random.sample(data.columns.tolist(), 2)
            self.data = data[random_columns]
            
        else:
            
            self.n_columns = len(col_names)
            self.data = data[col_names]
                
        # Scale the data
        self.scaling_method(scaling_method)
        
        if plot_data and self.n_columns == 2:
        
            # Plot initial data
            self._plot_initial_data()
            
        if plot_data and self.n_columns == 3:
            
            self._plot_initial_data_3d()
            
            
        # Get initial centroids
        centroids = self.__method()
        
        prev_clusters = np.zeros(len(self.data))
        no_change_count = 0
        
        while self.iteration <= self.max_iter and no_change_count < self.patience:
            
            # Assign each data point to the closest centroid
            cluster_array = self._add_data_point_to_cluster(centroids)
            
            # Check for convergence
            if np.array_equal(prev_clusters, cluster_array):
            
                no_change_count += 1
            
            else:
            
                no_change_count = 0
            
            prev_clusters = cluster_array.copy()
            
            if plot_data and self.n_columns == 2:
                # Plot current state
                self._update_plot(centroids, cluster_array)
                
            if plot_data and self.n_columns == 3:
                # Plot current state
                self._update_plot_3d(centroids, cluster_array)
            
            print("Cluster Array", cluster_array)
            
            # Update centroids
            centroids = self._update_centroids(cluster_array)
            
            print("Updated", centroids)
            
            self.iteration += 1
        
        if plot_data and self.n_columns == 2:
            
            plt.show()
            
        elif plot_data and self.n_columns == 3:
            
            self.create_gif()
        
        # Save the final state
        self.cluster_array = cluster_array
    
    ############################################################################################################
    ############################################# Scaling Functions ############################################
    ############################################################################################################
    def scaling_method(self, method):
        
        if method == 'min_max':
            
            self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())
            # print(method)
            # print(self.data)
            
        elif method == 'standardization':
                
            self.data = (self.data - self.data.mean()) / self.data.std()
            # print(method)
            # print(self.data)
            
        elif method == 'mean_normalization':
            
            self.data = (self.data - self.data.mean()) / (self.data.max() - self.data.min())
            # print(method)
            # print(self.data)
            
        else:
                
            raise ValueError(f"Unsupported scaling method: {method}, supported methods are: min_max, standardization, mean_normalization")

    
    ############################################################################################################
    ############################################ Centroid Functions ############################################
    ############################################################################################################
    
    def __method(self):
        
        if self.method == 'kmeans++':

            return self.__k_means_plus_plus()
        
        elif self.method == 'random':
            
            return self.__random_centroids()
        
        else:
                
            raise ValueError(f"Unsupported method: {self.method}, supported methods are: kmeans++, random")
        
    def __k_means_plus_plus(self):
        
        centroids = self.data.iloc[np.random.choice(len(self.data), 1, replace=False)].values
        
        if self.__show_initial_centroids:
            self.__update_initial_centroids(centroids)
        
        for i in range(self.k - 1):
            
            ## initialize a list to store distances of data points from nearest centroid
            dist = []

            for i in range(self.data.shape[0]):
                
                point = self.data.iloc[i, :].values
                d = np.inf
                
                ## compute distance of 'point' from each of the previously selected centroid and store the minimum distance
                for j in range(len(centroids)):
                        
                    temp_dist = self.__get_distance(point, centroids[j])
                    d = min(d, temp_dist)
                
                dist.append(d)
                
            ## select data point with maximum distance as our next centroid
            centroids = np.append(centroids, [self.data.iloc[np.argmax(np.array(dist)), :]], axis = 0)

            if self.__show_initial_centroids:
                self.__update_initial_centroids(centroids)
        
        return centroids
      
    def __random_centroids(self):
        
        centroids = self.data.iloc[np.random.choice(len(self.data), self.k, replace=False)].values
        
        if self.__show_initial_centroids:
                self.__update_initial_centroids(centroids)
        
        return centroids
    
    def _update_centroids(self, clusters):
        
        new_centroids = np.zeros((self.k, self.n_columns))
        
        for i in range(self.k):
            
            new_centroids[i] = np.mean(self.data[clusters == i], axis = 0)
        
        return new_centroids
    
    def _add_data_point_to_cluster(self, centroids):
        
        distances = np.zeros((len(self.data), self.k))
        
        for i, centroid in enumerate(centroids):

            distances[:, i] = self.__get_distance(self.data.values, centroid)
        
        return np.argmin(distances, axis=1)
        
    def __get_distance(self, points, centroid):
        
        if self.distance_metric == 'euclidean':

            return np.sqrt(np.sum(np.square(points - centroid), axis = 0 if len(points) == 2 else 1)) 

        elif self.distance_metric == 'manhattan':
            
            # Compute the Manhattan distance
            return np.sum(np.abs(points - centroid), axis=1)
        
        elif self.distance_metric == 'squared_euclidean':
            
            return np.sum(np.square(points - centroid), axis = 1)
            
        elif self.distance_metric == 'canberra':
            
            return np.sum(np.abs(points - centroid) / (np.abs(points) + np.abs(centroid)), axis=1)
            
        else:
            
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        
    ############################################################################################################
    ############################################ Plotting Functions ############################################
    ############################################################################################################
    
    def _plot_initial_data(self):
        
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.ax.scatter(self.data.iloc[:, 0], self.data.iloc[:, 1], marker='o', label='Data Points')
        self.ax.set_title('Initial State')
        self._plot_attributes()
        plt.tight_layout()
        plt.show(block=False)  # Display the plot without blocking
        plt.waitforbuttonpress(0)

    def _update_plot(self, centroids, clusters):
        
        self.ax.clear()  # Clear the axes
        
        print(centroids)
        
        # Use a colormap to get distinct colors for each cluster
        colors = cm.rainbow(np.linspace(0, 1, self.k))

        for i in range(self.k):
            
            cluster_points = self.data[clusters == i]
            
            self.ax.scatter(cluster_points.iloc[:, 0], cluster_points.iloc[:, 1], marker='o', label=f'Cluster {i}', color=colors[i])

            # Scatter plot for centroids with the color of the corresponding cluster
            self.ax.scatter(centroids[i, 0], centroids[i, 1], marker='*', s=200, c=colors[i], edgecolors='black', label=f'Centroid {i}')

        
        self.ax.set_title(f'Current State Iteration: {self.iteration}')
        self._plot_attributes()
        self._plot_layout()
        
    def _plot_attributes(self):

        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        self.ax.set_xlabel(self.data.columns[0])
        self.ax.set_ylabel(self.data.columns[1])

    def _plot_layout(self):
        
        plt.tight_layout()
        plt.show(block=False)  # Display the plot without blocking
        plt.pause(1)
        
    # function to plot the selected centroids
    def __update_initial_centroids(self, centroids):
        
        self.ax.clear()  # Clear the axes

        # Function to check if a row is equal to the centroid
        def is_equal_to_centroids(row):
            return any(np.all(row.values == centroid) for centroid in centroids)

        # Apply the function to identify rows equal to the centroid
        mask = self.data.apply(is_equal_to_centroids, axis=1)
        # Filter the DataFrame to exclude rows equal to the centroid
        result_df = self.data[~mask]

        self.ax.scatter(result_df.iloc[:, 0], result_df.iloc[:, 1], marker='o', label='Data Points')
        
        if self.method == "kmeans++":
            
            self.ax.scatter(centroids[:-1, 0], centroids[:-1, 1], marker='^', s=150, color = 'orange', label = 'Previously selected centroids')
            self.ax.scatter(centroids[-1, 0], centroids[-1, 1], marker='*', s=150, color = 'red', label = 'Next centroid')
        
        else:
            self.ax.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=150, color = 'orange', label = 'Next centroid')
        
        self.ax.set_title('Select % d th centroid'%(centroids.shape[0]))
        self._plot_attributes()
        self._plot_layout()

    ############################################################################################################
    ########################################## Plotting Functions 3d ###########################################
    ############################################################################################################ 
        
    def _plot_initial_data_3d(self):
        
        self.fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
        scatter = go.Scatter3d(
            x=self.data.iloc[:, 0],
            y=self.data.iloc[:, 1],
            z=self.data.iloc[:, 2],
            mode='markers',
            marker=dict(size=5, color='blue'),
            name='Data Points'
        )
        self.fig.add_trace(scatter)
        self.fig.update_layout(scene=dict(zaxis=dict(showticklabels=False)))
        self.fig.update_layout(title_text='Initial State')
        self._plot_attributes_3d()
        self.fig.to_image(f'./3d_images/plot_3d_iteration_initial.jpeg')


    def _update_plot_3d(self, centroids, clusters):
        
        self.fig.data = []  # Clear previous traces
        colors = px.colors.qualitative.G10

        for i in range(self.k):
            cluster_points = self.data[clusters == i]
            scatter_cluster = go.Scatter3d(
                x=cluster_points.iloc[:, 0],
                y=cluster_points.iloc[:, 1],
                z=cluster_points.iloc[:, 2],
                mode='markers',
                marker=dict(size=5, color=colors[i]),
                name=f'Cluster {i}'
            )
            self.fig.add_trace(scatter_cluster)

            scatter_centroid = go.Scatter3d(
                x=[centroids[i, 0]],
                y=[centroids[i, 1]],
                z=[centroids[i, 2]],
                mode='markers',
                marker=dict(size=8, color=colors[i], symbol='diamond', line = dict(color='black', width = 2)),
                name=f'Centroid {i}'
            )
            self.fig.add_trace(scatter_centroid)

        self.fig.update_layout(scene=dict(zaxis=dict(showticklabels=False)))
        self.fig.update_layout(title_text=f'Current State Iteration: {self.iteration}')
        self._plot_attributes_3d()

    def _plot_attributes_3d(self):
        
        layout = go.Layout(
                            scene = dict(
                                    xaxis = dict(title  = self.data.columns[0]),
                                    yaxis = dict(title  = self.data.columns[1]),
                                    zaxis = dict(title  = self.data.columns[2])
                                )
                          )
        self.fig.update_layout(layout)
        
        print(f"Current State Iteration: {self.iteration}")
        
        # Save the current figure as bytes
        img_bytes = self.fig.to_image(format="png", engine="kaleido")

        # Append the image bytes to a list
        if not hasattr(self, 'image_bytes_list'):
            self.image_bytes_list = []
            
        self.image_bytes_list.append(img_bytes)
        
    def save_gif(self, gif_path):
        # Create a GIF from the stored image bytes
        images = [Image.open(BytesIO(img_bytes)) for img_bytes in self.image_bytes_list]
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=500, loop=0)




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

        # print("=" * 30, "Data Overview", "=" * 30, "\n")

        # print(df_customers.head(), "\n")
        # print(df_customers.info(), "\n")
        # print(df_customers.describe(), "\n")

        return df_customers
    
    df_customers = read_present_data()

    # data_analysis(df_customers)
    # display(df_customers)

    kmeans = KMeans(k = 4, max_iter = 5, method = 'random', show_initial_centroids=True, distance_metric='euclidean', random_state = 42)
    kmeans.fit(df_customers, col_names = ['Age', 'Number of Purchases'], scaling_method='standardization', plot_data = True)#, 'Average Purchase Amount', 'Spending Score (1-100)'])

