import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial

# Assuming these are in your utils.py file
from utils import (
    wordToSequence,
    generate_classification_labels,
    german_characters,
    split_data,
)


def load_data_from_csv(filename):
    return pd.read_csv(filename)


def crop_word(word, max_length):
    return word[-max_length:]


def preprocessing(chunk, max_characters):
    chunk.dropna(inplace=True)
    chunk['word'] = chunk['word'].str.lower()
    chunk['word'] = chunk['word'].apply(lambda x: crop_word(x, max_characters))
    chunk['sequence'] = chunk['word'].map(
        partial(
            wordToSequence,
            possible_characters=german_characters,
            set_length=max_characters,
        )
    )
    chunk['labels'] = chunk['gender'].map(
        partial(generate_classification_labels, classes=possible_classes)
    )
    chunk['labels'] = chunk['labels'].map(
        lambda x: [item for sublist in x for item in sublist]
    )
    return chunk


def x_to_array(x, column):
    array = np.array(x[column].tolist())
    return tf.cast(array, dtype=tf.float32)


# Load and preprocess the data
max_characters = 7
possible_classes = ['m', 'f', 'n']

dataset = load_data_from_csv('dataset.csv')
dataset = preprocessing(dataset, max_characters)

# Split the data
_, _, _, _, test_x, test_y = split_data(
    dataset, ['sequence'], ['labels'], test_size=1, train_size=0, val_size=0
)

# Convert to arrays
test_x = x_to_array(test_x, 'sequence')
test_y = x_to_array(test_y, 'labels')

# Load the trained model
model = keras.models.load_model('models/model_lstm4.h5')

# Make predictions
y_pred = model.predict(test_x)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(test_y, axis=1)

# Generate confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=possible_classes,
    yticklabels=possible_classes,
)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.close()

# Calculate and print accuracy
accuracy = np.sum(y_pred_classes == y_true_classes) / len(y_true_classes)
print(f"Test Accuracy: {accuracy:.4f}")


# Accuracy, Precision, and Recall calculation
def calculate_metrics(cm):
    # True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN)
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    tn = np.sum(cm) - (fp + fn + tp)

    # Accuracy
    accuracy = np.sum(tp + tn) / np.sum(cm)

    # Precision
    precision = np.sum(tp) / np.sum(tp + fp)

    # Recall
    recall = np.sum(tp) / np.sum(tp + fn)

    return accuracy, precision, recall


# Calculate metrics from confusion matrix
accuracy, precision, recall = calculate_metrics(cm)

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Print classification report
from sklearn.metrics import classification_report

print(
    classification_report(y_true_classes, y_pred_classes, target_names=possible_classes)
)
