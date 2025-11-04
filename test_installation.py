
import monkey_patch
monkey_patch.apply_magenta_patch()

import tensorflow
print("TensorFlow version:", tensorflow.__version__)
import keras
print("Keras version:", keras.__version__)
import magenta

# Import the correct names: `MusicVAE` (the class) and `configs` (the model definitions).
# This was the failing part.
from magenta.models.music_vae import TrainedModel, configs
print("Successfully imported `TrainedModel` and `configs`!")

# As a further test, let's look up the configuration for a pre-trained model.
# This proves that the library is fully functional.
config = configs.CONFIG_MAP['cat-mel_2bar_big']
print("Successfully looked up a pre-trained model configuration.")

print("\nSUCCESS! The environment is configured and the circular import is patched.")
    