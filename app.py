import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the trained English-to-French model
@st.cache_resource()
def load_models():
    model = keras.models.load_model("fren_to_eng.keras")

    # Restore encoder
    encoder_inputs = model.input[0]
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = keras.Model(encoder_inputs, encoder_states)

    # Restore decoder
    decoder_inputs = model.input[1]
    decoder_state_input_h = keras.Input(shape=(256,), name="input_3")
    decoder_state_input_c = keras.Input(shape=(256,), name="input_4")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h_dec, state_c_dec]

    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = keras.Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
    )

    return encoder_model, decoder_model

encoder_model, decoder_model = load_models()

# Load token mappings
input_token_index = np.load("input_token_index.npy", allow_pickle=True).item()
target_token_index = np.load("target_token_index.npy", allow_pickle=True).item()
reverse_target_char_index = {i: char for char, i in target_token_index.items()}

# Translation function
def decode_sequence(input_text):
    input_seq = np.zeros((1, max_encoder_seq_length, len(input_token_index)), dtype="float32")

    # Convert input text into numerical sequence
    for t, char in enumerate(input_text):
        if char in input_token_index:
            input_seq[0, t, input_token_index[char]] = 1.0

    # Encode the input text
    states_value = encoder_model.predict(input_seq)

    # Initialize target sequence with start token "\t"
    target_seq = np.zeros((1, 1, len(target_token_index)))
    target_seq[0, 0, target_token_index["\t"]] = 1.0

    decoded_sentence = ""
    stop_condition = False

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Get the most likely character
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index.get(sampled_token_index, "")

        decoded_sentence += sampled_char

        # Stop if end token "\n" is found
        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # Update target sequence with the predicted character
        target_seq = np.zeros((1, 1, len(target_token_index)))
        target_seq[0, 0, sampled_token_index] = 1.0

        # Update states
        states_value = [h, c]

    return decoded_sentence.strip()


# Streamlit UI
st.title("English to French Translator 🇬🇧➡️🇫🇷")
st.write("Translate English sentences to French using a trained neural machine translation model.")

input_text = st.text_input("Enter an English sentence:", "")

if st.button("Translate"):
    if input_text:
        translation = decode_sequence(input_text)
        st.success(f"**French Translation:** {translation}")
    else:
        st.warning("Please enter a sentence.")
