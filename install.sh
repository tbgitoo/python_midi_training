python -m venv .venv

.\.venv\Scripts\activate

pip install numpy==1.26.4




# Upgrade pip to ensure compatibility
pip install --upgrade pip

# Install TensorFlow, the core ML framework
pip install tensorflow

# Install Magenta, which contains MusicVAE and other music utilities. [1, 19]
pip install magenta

# Install pretty_midi for parsing and handling MIDI files easily. [10, 18]
pip install pretty_midi

# Install a vector search library. faiss-cpu is a great starting point.
# It's CPU-only and easier to install than the GPU version. [4, 15]
pip install faiss-cpu

# Jupyter for running notebook-style experiments in VS Code
pip install jupyter
