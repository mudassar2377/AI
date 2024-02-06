import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

model = tf.keras.models.load_model("model.h5")
x_test = np.load("test_data.npy")
y_test = np.load("test_labels.npy")
x_test = x_test / 255.0

# Get predictions
predictions = model.predict(x_test)

# convert labels back from one hot encoded form
predicted_labels = np.argmax(predictions, axis=1)

confusion_mtx = tf.math.confusion_matrix(y_test, predicted_labels,name = 'confusion_matrix')

# Display the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues', )
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.legend('0: Dogs, 1: Cats')
plt.show()