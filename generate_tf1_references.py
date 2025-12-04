# generate_tf1_references.py
import tensorflow.compat.v1 as tf
import numpy as np
import os

# Correctly import the necessary modules from the Magenta library
from magenta.models.music_vae import model as music_vae_model_lib
from magenta.models.music_vae import configs
from magenta.contrib import training as contrib_training
from magenta.models.music_vae.data import OneHotMelodyConverter

# --- Main Configuration ---
CHECKPOINT_PATH = "models/download.magenta.tensorflow.org/models/music_vae/checkpoints/mel_2bar_big.ckpt"
MODEL_NAME = 'cat-mel_2bar_big'
OUTPUT_FILENAME = "tf1_reference_tensors.npz"

def generate_references():
    """
    Builds the TF1 MusicVAE graph using Magenta's core logic, loads a checkpoint,
    runs a teacher-forcing pass, and saves intermediate tensors for comparison.
    This script mimics the setup of `TrainedModel` but builds the training graph.
    """
    print("--- Generating TF1 Reference Tensors (Hybrid Approach) ---")

    # 1. Disable TF2 behavior to ensure the TF1 graph runs correctly.
    tf.disable_v2_behavior()

    # 2. Load the specified model configuration from Magenta.
    try:
        config = configs.CONFIG_MAP[MODEL_NAME]
    except KeyError:
        print(f"ERROR: Model configuration '{MODEL_NAME}' not found.")
        return

    latent_dim = config.hparams.z_size
    sequence_length = 32
    batch_size = 1
    converter = OneHotMelodyConverter(
        min_pitch=config.data_converter.min_pitch,
        max_pitch=config.data_converter.max_pitch,
        num_velocity_bins=config.data_converter.num_velocity_bins)
    output_depth = converter.output_depth

    # 3. Create deterministic inputs for reproducibility.
    np.random.seed(42)
    z_np = np.random.randn(batch_size, latent_dim).astype(np.float32)
    teacher_input_np = np.random.rand(batch_size, sequence_length, output_depth).astype(np.float32)

    final_references = {}

    # We run the graph multiple times with progressively deeper models
    # to safely fetch the output of each intermediate layer.
    for num_layers_to_run in range(1, len(config.hparams.dec_rnn_size) + 1):
        with tf.Graph().as_default(), tf.Session() as sess:
            print(f"\n--- Running TF1 model with {num_layers_to_run} decoder layer(s) ---")

            # Temporarily modify hparams to build a shallower model for this run.
            hparams_copy = contrib_training.HParams(**config.hparams.values())
            hparams_copy.dec_rnn_size = config.hparams.dec_rnn_size[:num_layers_to_run]

            # This is the key insight from `TrainedModel`: the config holds the model class.
            model = config.model
            # CRITICAL: We build with `is_training=True` to get the teacher-forcing graph.
            model.build(hparams_copy, output_depth, is_training=True)

            # --- THE FIX: Explicitly trigger the graph construction ---
            # The `build` method only sets up the layers. We must call a method
            # that uses them (like `reconstruct`) to force TF to create the ops.
            # We create placeholders that this method will use.
            z_input_placeholder = tf.placeholder(
                tf.float32, shape=[None, hparams_copy.z_size])
            targets_placeholder = tf.placeholder(
                tf.float32, shape=[None, None, output_depth])
            lengths_placeholder = tf.placeholder(
                tf.int32, shape=[None])
            
            # --- THE FIX: Call the correct method to build the teacher-forcing graph ---
            # The `reconstruction_loss` method on the decoder object is what triggers
            # the creation of the teacher-forcing forward pass in the TF1 graph.
            model.decoder.reconstruction_loss(targets_placeholder, lengths_placeholder, z_input_placeholder, model.encoder)

            # Restore the weights into the graph.
            saver = tf.train.Saver()
            saver.restore(sess, CHECKPOINT_PATH)

            # Get input placeholders from the graph by name.
            graph = sess.graph
            z_input = graph.get_tensor_by_name('z:0')
            targets_input = graph.get_tensor_by_name('targets:0')
            go_input = graph.get_tensor_by_name('go:0')
            lengths_input = graph.get_tensor_by_name('lengths:0')

            feed_dict = {
                z_input: z_np,
                targets_input: teacher_input_np,
                go_input: teacher_input_np[:, 0, :],
                lengths_input: [sequence_length] * batch_size,
            }

            # Define tensors to fetch for this specific run.
            tensors_to_fetch = {}
            if num_layers_to_run == 1:
                tensors_to_fetch["initial_state_flat"] = "decoder/z_to_initial_state/BiasAdd:0"

            # The final output of the current RNN stack is the output of this cell.
            tensors_to_fetch[f"cell_{num_layers_to_run-1}_output"] = model.decoder_output

            # The final logits are only available when running the full model.
            if num_layers_to_run == len(config.hparams.dec_rnn_size):
                tensors_to_fetch["final_logits"] = model.logits

            # Get graph tensors by name for any string-based names.
            tensors_to_fetch = {k: (v if not isinstance(v, str) else graph.get_tensor_by_name(v))
                                for k, v in tensors_to_fetch.items()}

            fetched_tensors = sess.run(tensors_to_fetch, feed_dict=feed_dict)

            # Process and store the results.
            for key, value in fetched_tensors.items():
                if 'cell' in key or 'final_logits' in key:
                    final_references[key] = value[0, 0, :] # Get first batch, first timestep
                else: # For initial_state_flat
                    final_references[key] = value[0]
                print(f"  - Stored '{key}' with shape {final_references[key].shape}")

    np.savez(OUTPUT_FILENAME, **final_references)
    print(f"\nSuccessfully saved all reference tensors to '{OUTPUT_FILENAME}'")

if __name__ == '__main__':
    if not os.path.exists(CHECKPOINT_PATH + ".index"):
        print(f"ERROR: Checkpoint file not found at '{CHECKPOINT_PATH}'. Please check the path.")
    else:
        generate_references()
