from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import RepeatVector
import numpy as np

# For creating an Encoder -
import tensorflow.keras as keras

# We need following for creating Encoder...
en_len = 15
en_vocab = 150
hsize = 48
# No matter how complex the model, the unederline concdept of encoder remains the same
#  ---------------- ENCODER -----------------------------

# Define an input layer
en_inputs = keras.layers.Input(shape=(en_len, en_vocab))

# Define a GRU layer which returns the state
en_gru = keras.layers.GRU(hsize, return_state=True)

# Get the output and state from the GRU
en_out, en_state = en_gru(en_inputs)

# Define and print the model summary
encoder = keras.models.Model(inputs=en_inputs, outputs=en_state)
print(encoder.summary())

# ---------------- DECODER ----------------------------------
from tensorflow.keras.layers import RepeatVector

hsize = 48
# average length of French sentences
fr_len = 20
# Define a RepeatVector layer
de_inputs = RepeatVector(fr_len)(en_state)
# Define a GRU model that returns all outputs
decoder_gru = GRU(hsize, return_sequences=True)
# Get the outputs of the decoder
gru_outputs = decoder_gru(de_inputs, initial_state=en_state)

# ENCODER <-> DECODER
# Define a model with the correct inputs and outputs
enc_dec = Model(inputs=en_inputs, outputs=gru_outputs)
print(enc_dec.summary())


