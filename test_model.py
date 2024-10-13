import tensorflow as tf
from tensorflow import keras
import numpy as np

# Import necessary functions from your utils module
from utils import wordToSequence, german_characters

# Load the saved model
model = keras.models.load_model('models/model_lstm4.h5')

# Constants
max_characters = 7
possible_classes = ['m', 'f', 'n']

def preprocess_word(word):
    word = word.lower()
    word = word[-max_characters:]  # Crop the word if it's longer than max_characters
    sequence = wordToSequence(word, possible_characters=german_characters, set_length=max_characters)
    return np.array([sequence])  # Return as a 2D array for model input

def predict_gender(word):
    input_sequence = preprocess_word(word)
    prediction = model.predict(input_sequence)[0]
    predicted_class = possible_classes[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

def main():
    print("Word Gender Predictor")
    print("Enter a word to predict its gender (or 'quit' to exit)")

    while True:
        word = input("\nEnter a word: ").strip()
        
        if word.lower() == 'quit':
            print("Goodbye!")
            break

        if not word:
            print("Please enter a valid word.")
            continue

        gender, confidence = predict_gender(word)
        print(f"Predicted gender: {gender}")
        print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()


#die Gabel, das Messer, der Löffel