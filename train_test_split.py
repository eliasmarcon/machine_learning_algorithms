import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# plot a sample of the data
def plot_sample(dataset):

    # get just the features of the dataframe and reshape them into a image format
    features = dataset.iloc[:, 1:].values.astype('int32').reshape(dataset.shape[0], 28, 28, 1)

    plt.figure(figsize = (10, 10))

    for i in range(1, 21):
        
        plt.subplot(5, 4, i)
        plt.title(dataset['label'][i])
        plt.imshow(features[i]) #cmap = plt.get_cmap('gray')
        plt.axis('off')
        
    plt.show()



def distribution_labels(target):

    # Count the occurrences of each label
    label_counts = np.bincount(target)

    # Create a bar plot with different colors and numbers on top of the bars
    plt.figure(figsize = (10, 6))
    bars = plt.bar(range(len(label_counts)), label_counts)

    # Add numbers on top of each bar
    for bar, count in zip(bars, label_counts):
        
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50, str(count), ha = 'center', fontsize = 10)

    plt.xlabel('Label')
    plt.ylabel('Count of Label')
    plt.title('Label Counts in Training Data without Augmentation')
    plt.xticks(range(len(label_counts)))

    # Assign different colors to each bar
    colors = plt.cm.viridis(np.linspace(0, 1, len(label_counts)))

    for bar, color in zip(bars, colors):

        bar.set_color(color)

    plt.show()


# create custom train test function
def custom_train_test_split(df, target_column, testing_percentage = 0.2, random_seed = None, num_shuffles = 100):
     
    # Check if the target column exists in the DataFrame
    if target_column not in df.columns:
    
        raise ValueError(f"Target column '{target_column}' not found in the DataFrame.")

    # set a seed if one is defined
    if random_seed is None: 
        random_seed = 42

    # check if shuffle 
    if num_shuffles < 0:
        
        # Determine the split index
        split_index = int(len(df) * testing_percentage)

        # Split the DataFrame into training and testing sets
        train_df = df[:split_index]
        test_df = df[split_index:]
    
    else:

        # Shuffle the DataFrame
        for _ in range(num_shuffles):
            
            train_df = df.sample(frac = 1- testing_percentage, random_state = random_seed).reset_index(drop = True)
            test_df = df.sample(frac = testing_percentage, random_state = random_seed).reset_index(drop = True)

    # Extract features and target variable
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    return X_train, X_test, y_train, y_test


def separator():

    print("\n\n")
    print("##############################################################################################################################")
    print("\n\n")

if __name__ == "__main__":

    # Import Dataset
    dataset = pd.read_csv("images.csv")
    dataset = dataset.astype(int)

    # example of the dataset
    print(dataset)

    # plot a sample of the data
    plot_sample(dataset)

    # plot distribution of labels
    distribution_labels(dataset['label'])

    # create self implemented train/test function
    X_train_custom, X_test_custom, y_train_custom, y_test_custom = custom_train_test_split(dataset, 'label', testing_percentage = 0.2, random_seed = 42, num_shuffles = 1)
    X_train_2_custom, X_test_2_custom, y_train_2_custom, y_test_2_custom = custom_train_test_split(dataset, 'label', testing_percentage = 0.9, random_seed = 42, num_shuffles = 1)

    
    separator()
    print(f"Self implemented split X_train_custom:\n {X_train_custom}\n")
    separator()
    print(f"Self implemented split y_train_custom:\n {y_train_custom.to_frame()}\n")
    separator()
    print(f"Self implemented split X_test_custom:\n {X_test_custom}\n")
    separator()
    print(f"Self implemented split y_test_custom:\n {y_test_custom.to_frame()}\n")
    

    #sklearn train_test_split
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(dataset.iloc[:, 1:], dataset['label'], test_size = 0.2, random_state = 42)
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(dataset.iloc[:, 1:], dataset['label'], test_size = 0.9, random_state = 42)

    separator()
    print(f"Sklearn split X_train_2:\n {X_train_1}\n")
    separator()
    print(f"Sklearn split y_train_2:\n {y_train_1.to_frame()}\n")
    separator()
    print(f"Sklearn split X_test_2:\n {X_test_2}\n")
    separator()
    print(f"Sklearn split y_test_2:\n {y_test_2.to_frame()}\n")

