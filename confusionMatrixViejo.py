import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial

# Assuming these are in your utils.py file
from utils import wordToSequence, generate_classification_labels, german_characters, split_data

def load_data_from_csv(filename):
    return pd.read_csv(filename)

def preprocessing(chunk, max_characters):
    chunk['word'] = chunk['word'].str.lower()
    chunk = chunk[chunk['word'].str.len() < max_characters]
    chunk.dropna(inplace=True)
    chunk['sequence'] = chunk['word'].map(partial(wordToSequence, possible_characters=german_characters, set_length=max_characters))
    chunk['labels'] = chunk['gender'].map(partial(generate_classification_labels, classes=possible_classes))
    chunk['labels'] = chunk['labels'].map(lambda x: [item for sublist in x for item in sublist])
    return chunk

def x_to_array(x, column):
    array = np.array(x[column].tolist())
    return tf.cast(array, dtype=tf.float32)

# Load and preprocess the data
max_characters = 20
possible_classes = ['m', 'f', 'n']

dataset = load_data_from_csv('dataset.csv')
dataset = preprocessing(dataset, max_characters)

# Split the data
_, _, _, _, test_x, test_y = split_data(dataset, ['sequence'], ['labels'])

# Convert to arrays
test_x = x_to_array(test_x, 'sequence')
test_y = x_to_array(test_y, 'labels')

# Recreate the model architecture
model = keras.models.Sequential()
model.add(layers.Embedding(len(german_characters), 32, input_length=max_characters))
model.add(layers.Bidirectional(layers.LSTM(128)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

# Load weights
model.load_weights('models/model_lstm3.h5')

# Compile the model (use the same loss and optimizer as in your training script)
model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.SGD(), metrics=['accuracy'])

# Make predictions
y_pred = model.predict(test_x)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(test_y, axis=1)

# Generate confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=possible_classes,
            yticklabels=possible_classes)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.close()

# Calculate and print accuracy
accuracy = np.sum(y_pred_classes == y_true_classes) / len(y_true_classes)
print(f"Test Accuracy: {accuracy:.4f}")

# Print classification report
print(classification_report(y_true_classes, y_pred_classes, target_names=possible_classes))