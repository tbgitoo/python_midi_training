import tensorflow as tf
from tensorflow.keras.layers import LSTMCell, RNN, Dense, StackedRNNCells
import os
import requests

# Helper class for the specific step logic in autoregressive generation
class AutoregressiveStep(tf.keras.layers.Layer):
    def __init__(self, lstm_cells, output_projection_layer, **kwargs):
        super(AutoregressiveStep, self).__init__(**kwargs)
        self.lstm_cells = lstm_cells
        self.output_projection = output_projection_layer
        # The state size is the sum of the state sizes of all LSTM cells
        self.state_size = [cell.state_size for cell in self.lstm_cells]

    def call(self, inputs, states):
        # `inputs` is now a single concatenated tensor for this timestep.
        # The concatenation of (step_input, z) happens *before* the RNN layer.

        # Manually run the stack of LSTM cells for a single step
        cell_input = inputs
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
    @staticmethod
    def get_default_config():
        """Returns the default configuration dictionary for the 'cat-mel_2bar_big' model."""
        return {
            "output_depth": 90,
            "latent_dim": 512,
            "lstm_units": 2048,
            "num_layers": 3,
            "sequence_length": 32,
        }

    def __init__(self, config=None, **kwargs):
        super(MusicVAEDecoder, self).__init__(**kwargs)

        # If no config is provided, use the default.
        if config is None:
            config = self.get_default_config()

        # --- Store all architectural parameters as instance attributes ---
        self.output_depth = config["output_depth"]
        self.latent_dim = config["latent_dim"]
        self.lstm_units = config["lstm_units"]
        self.num_layers = config["num_layers"]
        self.sequence_length = config["sequence_length"]
        self.vocab_size = self.output_depth # Alias for clarity

        self.z_to_initial_state = Dense(self.lstm_units * self.num_layers * 2, name="z_to_initial_state")
        self.z_to_initial_state.build(input_shape=(None, self.latent_dim)) # Explicitly build with known latent_dim
        self.lstm_cells = [LSTMCell(self.lstm_units, name=f"lstm_cell_{i}") for i in range(self.num_layers)]
        stacked_cells = StackedRNNCells(self.lstm_cells)
        self.rnn = RNN(stacked_cells, return_sequences=True, return_state=True, name="decoder_rnn")
        self.output_projection = Dense(self.output_depth, name="output_projection")


        # --- THE FIX: Create the autoregressive step layer ---
        autoregressive_step = AutoregressiveStep(self.lstm_cells, self.output_projection)
        # --- The RNN wrapper for generation ---
        self.generation_rnn = tf.keras.layers.RNN(autoregressive_step, return_sequences=True)

    


    # --- 2. The "Teaching" Endpoint: Decorated for Training/Reconstruction ---
    # This will be one of the functions available in your saved model.
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], name="z"),
        tf.TensorSpec(shape=[None, None, None], name="inputs")
    ])
    def reconstruct(self, z, inputs):
        """
        Performs the forward pass of the decoder.
        """
        # 1. Get initial state from z
        initial_state=self._get_initial_state(z)

    
        # 2. Prepare the latent vector, making sure it matches the input's dynamic sequence length.
        # THE FIX: Get the sequence length dynamically from the `inputs` tensor.
        current_sequence_length = tf.shape(inputs)[1]
        z_repeated = tf.tile(tf.expand_dims(z, 1), [1, current_sequence_length, 1])
        
        
        # 3. Concatenate z with the inputs along the feature dimension.
        rnn_inputs = tf.concat([inputs, z_repeated], axis=-1)

         # Run the RNN.
        rnn_output, *_ = self.rnn(rnn_inputs, initial_state=initial_state)

        # Project the RNN output to the final output space.
        output = self.output_projection(rnn_output)
        return output

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], name="z")
    ])
    def generate(self, z):
        batch_size = tf.shape(z)[0]
        
        # 1. Get initial state from z
        initial_state = self._get_initial_state(z) # Same initial state processing as for reconstruction
        
        # 2. Create the inputs for the RNN layer.
        # The RNN layer needs a sequence to iterate over. We give it a dummy sequence.
        # The actual input for each step will be constructed *inside* the AutoregressiveStep layer.
        dummy_sequence = tf.zeros([batch_size, self.sequence_length, self.vocab_size])
        
        # We also need to pass `z` as a constant to each step. We concatenate it
        # with the dummy sequence to create a single input tensor for the RNN.
        z_repeated = tf.tile(tf.expand_dims(z, 1), [1, self.sequence_length, 1])
        rnn_inputs = tf.concat([dummy_sequence, z_repeated], axis=-1)
        
        # 3. Call the generation RNN
        # This is now a single, clean, graph-native call.
        logits_sequence = self.generation_rnn(rnn_inputs, initial_state=initial_state)
        
        return logits_sequence
    
    def _get_initial_state(self, z):

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
    

    def call(self, inputs, training=False):
        """
        The primary forward pass for training (teacher-forcing).
        This ensures the `self.rnn` layer is part of the main model graph.
        `inputs` is expected to be a tuple: (z, teacher_sequence).
        """
        z, teacher_sequence = inputs
        # Delegate to the `reconstruct` logic.
        return self.reconstruct(z, teacher_sequence)
    
    def get_config(self):
        # This allows Keras to save the model's high-level configuration.
        config = super().get_config()
        config.update({
            "output_depth": self.output_depth,
            "latent_dim": self.latent_dim,
            "lstm_units": self.lstm_units,
            "num_layers": self.num_layers,
            "sequence_length": self.sequence_length,
        })
        return config

    @classmethod
    def from_config(cls, config):
        # This tells Keras how to create the model from the config.
        return cls(config=config)



   
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

        # --- WEIGHT RE-ORDERING AND BIAS ADJUSTMENT LOGIC ---
        def reorder_lstm_weights(kernel, bias, num_units):
            # TF1 format: [i, c, f, o] (input, cell, forget, output)
            # TF2 format: [i, f, c, o] (input, forget, cell, output)
    
            # Split into the 4 gate weights
            # The last dimension is 4 * num_units, so we split into 4 equal parts
            k_i, k_c, k_f, k_o = tf.split(kernel, 4, axis=-1)
            b_i, b_c, b_f, b_o = tf.split(bias, 4, axis=-1)

            # The legacy LSTMCell adds a `forget_bias` of 1.0 by default.
            # We must manually add it to the forget gate's bias.
            b_f = b_f + 1.0
    
            # Re-assemble in TF2 order (swap c and f)
            reordered_kernel = tf.concat([k_i, k_f, k_c, k_o], axis=-1)
            reordered_bias = tf.concat([b_i, b_f, b_c, b_o], axis=-1)
    
            return reordered_kernel, reordered_bias

        # Apply reordering and bias adjustment
        tf1_kernel, tf1_bias = reorder_lstm_weights(tf1_kernel, tf1_bias, cell.units)


        # THE FIX: Use the correct input dimension for splitting the kernel,
        # based on the layer index.
        if i == 0:
            # Input is teacher sequence (output_depth) + latent vector (latent_dim)
            input_dim = decoder_model.output_depth + decoder_model.latent_dim
        else:
            # Subsequent layers take the output of the previous LSTM layer.
            input_dim = cell.units

        # Perform the split at the correct index.
        keras_kernel = tf1_kernel[:input_dim, :]
        keras_recurrent_kernel = tf1_kernel[input_dim:, :]
        
        # Now the shapes and internal gate order will match perfectly.
        cell.set_weights([keras_kernel, keras_recurrent_kernel, tf1_bias])
        print(f"Loaded weights for LSTM cell {i} from '{tf1_kernel_name}'.")

    # --- 3. Load output_projection weights ---
    out_kernel = reader.get_tensor("decoder/output_projection/kernel")
    out_bias = reader.get_tensor("decoder/output_projection/bias")
    decoder_model.output_projection.set_weights([out_kernel, out_bias])
    print("Loaded weights for 'output_projection' layer.")

    print("\nSuccessfully loaded all decoder weights from Magenta checkpoint!")
