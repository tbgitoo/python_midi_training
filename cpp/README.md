# How to build the C++ encoder

This document provides instructions on how to build the C++ encoder application.

## Prerequisites

1.  **C++ Compiler:** You need a C++ compiler that supports C++11. On Windows, you can use Visual Studio. On Linux, you can use g++.
2.  **CMake:** You need to have CMake installed. You can download it from [https://cmake.org/](https://cmake.org/).
3.  **TensorFlow Source Code:** You need to have the TensorFlow source code cloned as a submodule in the project root.

    ```bash
    git submodule add https://github.com/tensorflow/tensorflow.git tensorflow
    ```

## Build Steps

1.  **Create a build directory:**

    ```bash
    mkdir build
    cd build
    ```

4.  **Run CMake:**

    ```bash
    cmake ..
    ```

5.  **Build the project:**

    On Windows, you can open the generated Visual Studio solution and build the `encoder` project. On Linux, you can run the following command:

    ```bash
    make
    ```

## Running the encoder

After building the project, you can run the encoder from the `build` directory:

```bash
./encoder <path_to_tflite_model>
```

For example:

```bash
./encoder ../../models/music_vae_encoder_tf2.tflite
```
