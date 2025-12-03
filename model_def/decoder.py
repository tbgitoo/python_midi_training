import tensorflow as tf
from tensorflow.keras.layers import LSTMCell, RNN, Dense
import os
import requests

LATENT_DIM = 512
OUTPUT_DEPTH = 90
SEQUENCE_LENGTH = 32 # The correct, fixed sequence length for this model
BATCH_SIZE = 1
LSTM_UNITS=2048
NUM_LAYERS=3

# Helper class for the specific step logic in autoregressive generation
class AutoregressiveStep(tf.keras.layers.Layer):
    def __init__(self, lstm_cells, output_projection_layer, **kwargs):
        super(AutoregressiveStep, self).__init__(**kwargs)
        self.lstm_cells = lstm_cells
        self.output_projection = output_projection_layer
        # The state size is the sum of the state sizes of all LSTM cells
        self.state_size = [cell.state_size for cell in self.lstm_cells]

    def call(self, inputs, states):
        # `inputs` is a tuple: (the input for this step, the constant z vector)
        step_input, z = inputs
        
        # Concatenate the current step's input with z
        step_input_with_z = tf.concat([step_input, z], axis=-1)

        # Manually run the stack of LSTM cells for a single step
        cell_input = step_input_with_z
        new_states = []
        for i, cell in enumerate(self.lstm_cells):
            cell_output, (new_h, new_c) = cell(cell_input, states=states[i])
            new_states.append([new_h, new_c])
            cell_input = cell_output
        
        final_cell_output = cell_output
        
        # Project to get the logits for this single step
        step_logits = self.output_projection(final_cell_output)
        
        # The output of this layer is the logits for the next step
        # The new state is passed internally by the RNN wrapper
        return step_logits, new_states
    
    def get_config(self):
        """
        Allows Keras to serialize the layer.
        
        We do NOT serialize the actual layers here because they are complex objects
        owned by the parent model. The parent model's config will handle them.
        This implementation relies on the parent model correctly reconstructing
        and passing these layers during its own `from_config` call.
        """
        config = super(AutoregressiveStep, self).get_config()
        # We don't need to add any special parameters to the config,
        # as the necessary layers will be passed to the constructor.
        return config



# The MusicVAEDecoder class remains the same
class MusicVAEDecoder(tf.keras.Model):
    """The decoder portion of the MusicVAE model."""
    def __init__(self, output_depth, lstm_units=2048, num_layers=3, name="decoder",sequence_length=32):
        super(MusicVAEDecoder, self).__init__(name=name)
        self.z_to_initial_state = Dense(lstm_units * num_layers * 2, name="z_to_initial_state")
        self.lstm_cells = [LSTMCell(lstm_units, name=f"lstm_cell_{i}") for i in range(num_layers)]
        self.rnn = RNN(self.lstm_cells, return_sequences=True, return_state=True, name="decoder_rnn")
        self.output_projection = Dense(output_depth, name="output_projection")
        self.vocab_size = output_depth
        self.sequence_length = sequence_length


        # --- THE FIX: Create the autoregressive step layer ---
        autoregressive_step = AutoregressiveStep(self.lstm_cells, self.output_projection)
        # --- The RNN wrapper for generation ---
        self.generation_rnn = tf.keras.layers.RNN(autoregressive_step, return_sequences=True)

    


    # --- 2. The "Teaching" Endpoint: Decorated for Training/Reconstruction ---
    # This will be one of the functions available in your saved model.
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, LATENT_DIM], name="z"),
        tf.TensorSpec(shape=[BATCH_SIZE, None, OUTPUT_DEPTH], name="inputs")
    ])
    def reconstruct(self, z, inputs):
        """
        Performs the forward pass of the decoder.
        """
        # 1. Get initial state from z
        initial_state=self.get_initial_state(z)

    
        # 2. Prepare the latent vector for concatenation at each time step.
        # We need to repeat `z` so it can be attached to every element of the sequence.
        # Tile z from shape [batch, latent_dim] to [batch, sequence_length, latent_dim]
        z_repeated = tf.tile(tf.expand_dims(z, 1), [1, self.sequence_length, 1])
        
        
        # Concatenate z with the inputs along the feature dimension.
        # `inputs` has shape [batch, sequence_length, 90]
            # `z_repeated` has shape [batch, sequence_length, 512]
            # The result will have shape [batch, sequence_length, 602]
        rnn_inputs = tf.concat([inputs, z_repeated], axis=-1)

         # Run the RNN.
        rnn_output, *_ = self.rnn(rnn_inputs, initial_state=initial_state)

        # Project the RNN output to the final output space.
        output = self.output_projection(rnn_output)
        return output

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, LATENT_DIM], name="z")
    ])
    def generate(self, z):
        batch_size = tf.shape(z)[0]
        
        # 1. Get initial state from z
        initial_state = self.get_initial_state(z) # Same initial state processing as for reconstruction
        
        # 2. Create the inputs for the RNN layer.
        # The RNN layer needs a sequence to iterate over. We give it a dummy sequence.
        # The actual input for each step will be constructed *inside* the AutoregressiveStep layer.
        dummy_sequence = tf.zeros([batch_size, self.sequence_length, self.vocab_size])
        
        # We also need to pass `z` as a constant to each step.
        z_repeated = tf.tile(tf.expand_dims(z, 1), [1, self.sequence_length, 1])
        
        # The RNN layer will unpack this tuple at each time step
        rnn_inputs = (dummy_sequence, z_repeated)
        
        # 3. Call the generation RNN
        # This is now a single, clean, graph-native call.
        logits_sequence = self.generation_rnn(rnn_inputs, initial_state=initial_state)
        
        return logits_sequence
    
    def get_initial_state(self, z):

        batch_size = tf.shape(z)[0]
        num_layers = len(self.lstm_cells)
        lstm_units = self.lstm_cells[0].units

        # Project the latent vector 'z' to get the initial state for the LSTM.
        # Shape: (batch_size, num_layers * 2 * lstm_units)
        initial_state_flat = self.z_to_initial_state(z)

        # Reshape to separate layers and the h/c states.
        # Shape: (batch_size, num_layers, 2, lstm_units)
        initial_state_reshaped = tf.reshape(
            initial_state_flat, [batch_size, num_layers, 2, lstm_units]
        )

        # Transpose to group h/c states by layer.
        # Shape: (num_layers, 2, batch_size, lstm_units)
        initial_state_transposed = tf.transpose(initial_state_reshaped, [1, 2, 0, 3])

        # Unstack to create the final list of states for each layer.
        # This creates a list of `num_layers` elements.
        # Each element is a tensor of shape (2, batch_size, lstm_units).
        initial_state_list = tf.unstack(initial_state_transposed)

        # Further unstack each layer's state into (h, c) tuples.
        # The final structure is: [ (h0, c0), (h1, c1), ... ]
        # which is what the Keras RNN layer expects.
        initial_state = [tf.unstack(s) for s in initial_state_list]

        return(initial_state)
    

    # --- THE FIX 1: Make `generate` logic the primary `call` method ---
    # This method is UNDECORATED.
    def call(self, z):
        """
        This is the primary forward pass, implementing autoregressive generation.
        """
        batch_size = tf.shape(z)[0]
        initial_state = self.get_initial_state(z)
        
        # The RNN layer needs a dummy sequence to know how many steps to run.
        dummy_sequence = tf.zeros([batch_size, self.sequence_length, self.vocab_size])
        
        # We also need to pass `z` as a constant to each step.
        z_repeated = tf.tile(tf.expand_dims(z, 1), [1, self.sequence_length, 1])
        
        rnn_inputs = (dummy_sequence, z_repeated)
        
        logits_sequence = self.generation_rnn(rnn_inputs, initial_state=initial_state)
        
        return logits_sequence
    
    def get_config(self):
        # This allows Keras to save the model's high-level configuration.
        return {
            "latent_dim": self.latent_dim,
            "lstm_units": self.lstm_units,
            "vocab_size": self.vocab_size,
            "sequence_length": self.sequence_length,
        }

    @classmethod
    def from_config(cls, config):
        # This tells Keras how to create the model from the config.
        return cls(**config)



   
def load_magenta_weights(decoder_model, checkpoint_path):
    """
    Loads weights from a TF1 Magenta checkpoint into a TF2 Keras decoder model.
    (Final corrected version)
    """
    reader = tf.train.load_checkpoint(checkpoint_path)

    # --- 1. Load z_to_initial_state weights ---
    z_kernel = reader.get_tensor("decoder/z_to_initial_state/kernel")
    z_bias = reader.get_tensor("decoder/z_to_initial_state/bias")
    decoder_model.z_to_initial_state.set_weights([z_kernel, z_bias])
    print("Loaded weights for 'z_to_initial_state' layer.")

    # --- 2. Load LSTM cell weights ---
    for i, cell in enumerate(decoder_model.lstm_cells):
        tf1_kernel_name = f"decoder/multi_rnn_cell/cell_{i}/lstm_cell/kernel"
        tf1_bias_name = f"decoder/multi_rnn_cell/cell_{i}/lstm_cell/bias"
        
        tf1_kernel = reader.get_tensor(tf1_kernel_name)
        tf1_bias = reader.get_tensor(tf1_bias_name)

        # THE FIX: Use the correct input dimension for splitting the kernel,
        # based on the layer index.
        if i == 0:
            # The original model's first layer has a complex input of dim 602.
            input_dim = 602
        else:
            # Subsequent layers take the output of the previous LSTM layer.
            input_dim = cell.units # which is 2048

        # Perform the split at the correct index.
        keras_kernel = tf1_kernel[:input_dim, :]
        keras_recurrent_kernel = tf1_kernel[input_dim:, :]
        
        # Now the shapes will match perfectly.
        cell.set_weights([keras_kernel, keras_recurrent_kernel, tf1_bias])
        print(f"Loaded weights for LSTM cell {i} from '{tf1_kernel_name}'.")

    # --- 3. Load output_projection weights ---
    out_kernel = reader.get_tensor("decoder/output_projection/kernel")
    out_bias = reader.get_tensor("decoder/output_projection/bias")
    decoder_model.output_projection.set_weights([out_kernel, out_bias])
    print("Loaded weights for 'output_projection' layer.")

    print("\nSuccessfully loaded all decoder weights from Magenta checkpoint!")












