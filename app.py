# 
import os # For checking if file exists
import matplotlib as plt # For plot
import numpy as np  # For saving/loading array data in .npz format
import tensorflow as tf  # To download the MNIST dataset


# =====================================================================================
# Load or Download and save dataset
DATA_FILE = 'dataset/mnist_keras_saved.npz'  # Define the filename to save/load MNIST data

# Check if the MNIST data file already exists locally
if not os.path.exists(DATA_FILE):
    # If file doesn't exist, download the dataset from TensorFlow
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Save the loaded data arrays into a compressed .npz file for future use
    np.savez(DATA_FILE, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    print("Downloaded and saved MNIST data.")
else:
    # If the file exists, load the dataset arrays directly from the saved .npz file
    with np.load(DATA_FILE) as data:
        x_train = data['x_train']  # Load training images
        y_train = data['y_train']  # Load training labels
        x_test = data['x_test']    # Load test images
        y_test = data['y_test']    # Load test labels
    print("Loaded MNIST data from local file.")

# Print the number of samples in training and test sets to verify loading
print(f"Train samples: {x_train.shape[0]}, Test samples: {x_test.shape[0]}")

