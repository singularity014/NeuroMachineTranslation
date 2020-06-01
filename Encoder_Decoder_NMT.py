from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import GRU

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
