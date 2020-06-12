from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import RepeatVector
# Import Dense and TimeDistributed layers
from tensorflow.keras.layers import Dense, TimeDistributed

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

# ---------------- Creating the later after decoder GRU output-------

# Define a softmax dense layer that has fr_vocab outputs
de_dense = Dense(fr_vocab, activation='softmax')
# Wrap the dense layer in a TimeDistributed layer
de_dense_time = TimeDistributed(de_dense)
# Get the final prediction of the model
de_pred = de_dense_time(de_out)
print("Prediction shape: ", de_pred.shape)

# ------------------ Compiling Model ----------------------------

# Define a model with encoder input and decoder output
nmt = Model(inputs=en_inputs, outputs=de_pred)

# Compile the model with an optimizer and a loss
nmt.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# View the summary of the model 
nmt.summary()
