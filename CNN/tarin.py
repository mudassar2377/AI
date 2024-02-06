import numpy as np
import tensorflow as tf
from os import system as sys

# Clear the terminal
sys("clear")

# Load the data
train_data = np.load("train_data.npy")
train_labels = np.load("train_labels.npy")
val_data = np.load("val_data.npy")
val_labels = np.load("val_labels.npy")

# Normalize the data
train_data = train_data / 255.0
val_data = val_data / 255.0
# Convert labels in one hot encoded form
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=2)
val_labels = tf.keras.utils.to_categorical(val_labels, num_classes=2)

# Load model
model = tf.keras.models.load_model("model.h5")
model.fit(train_data, train_labels,batch_size = 16 ,epochs=25, validation_data=(val_data, val_labels))
model.save("model.h5")
