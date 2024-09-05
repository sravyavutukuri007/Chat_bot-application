import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import json
import pickle

# Load the trained model
model = load_model('chat_model.h5')

# Load tokenizer and label encoder
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

# Load intents for responses
with open('intents.json') as file:
    data = json.load(file)

# Function to predict the intent and respond
def chat():
    print("Start talking with the bot (type 'quit' to stop)!")
    while True:
        input_text = input("You: ")
        if input_text.lower() == "quit":
            break

        # Convert input to a sequence and pad it
        sequences = tokenizer.texts_to_sequences([input_text])
        padded_sequences = pad_sequences(sequences, truncating='post', maxlen=20)

        # Predict the intent
        prediction = model.predict(padded_sequences)
        predicted_label = lbl_encoder.inverse_transform([np.argmax(prediction)])

        # Find the appropriate response
        for intent in data['intents']:
            if intent['tag'] == predicted_label[0]:
                print(f"Bot: {np.random.choice(intent['responses'])}")

# Start chatting with the bot
chat()
