import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import History

german_characters = "abcdefghijklmnopqrstuvwxyzäöüß"


def one_hot_word(word, possible_characters = "abcdefghijklmnopqrstuvwxyz", default_length = None):
    assert not (default_length != None and default_length < len(word))
    array = np.zeros((len(possible_characters) + 1, (len(word)+1) if default_length == None else default_length), dtype=np.int8)
    i = -1
    for i, char in enumerate(word):
        array[possible_characters.index(char), i] = 1
    array[len(possible_characters), i+1] = 1 #end of word

    return array

def split_data(dataset: pd.DataFrame, features: list[str], labels: list[str], train_size = 0.8, val_size = 0.1, test_size = 0.1):
    assert train_size + val_size + test_size == 1
    train = dataset.sample(frac=train_size, random_state=0)
    val_test = dataset.drop(train.index)
    val = val_test.sample(frac=val_size/(val_size + test_size), random_state=0)
    test = val_test.drop(val.index)
    return train[features], train[labels], val[features], val[labels], test[features], test[labels]


def one_hot_array(length: int, index: int):
    assert index < length and length > 0
    array = np.zeros((length), dtype=np.int8)
    array[index] = 1
    return array

def generate_classification_labels(labels: list[str], classes: list[str]) -> np.ndarray:
    return np.array([one_hot_array(len(classes), classes.index(label)) for label in labels])


def plot_accuracy(history: dict[str, list[float]], offset = 0):
    fig, ax = plt.subplots()
    xs = np.arange(len(history['accuracy'])) + offset
    ax.plot(xs, history['accuracy'])
    ax.plot(xs, history['val_accuracy'])
    ax.set_title('model accuracy')
    ax.set_ylabel('accuracy')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'validation'], loc='upper left')
    ax.grid()
    return fig

def merge_histories(histories: list[History]):
    merged_history = {}
    for history in histories:
        for key in history.history:
            if key in merged_history:
                merged_history[key] += history.history[key][:]
            else:
                merged_history[key] = history.history[key][:]
    return merged_history

def slice_history(history: dict[str, list[float]], start = None, end = None):
    sliced_history = {}
    for key in history:
        sliced_history[key] = history[key][start:end]
    return sliced_history

def wordToSequence(word: str, possible_characters = "abcdefghijklmnopqrstuvwxyz", set_length = None):
    array = np.array([possible_characters.index(char) for char in word])
    if set_length != None:
        array = np.pad(array, (0, set_length - len(word)))
    return array
