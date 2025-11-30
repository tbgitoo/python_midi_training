#   !/bin/bash
# Install required Python packages for the project
# This for python 3.10.0 

pip install tensorflow==2.12.0  note-seq pretty_midi faiss-cpu jupyter

pip install tensorflow-probability==0.19.0

pip install --no-deps magenta

pip install tf-slim

pip install tensorflow_datasets

pip uninstall protobuf 
    
pip install protobuf==3.20.3
# This specific version is required to avoid compatibility issues with magenta 


# For working with R scripts

pip install pyreadr

pip install radian
    
    
    
    




    

