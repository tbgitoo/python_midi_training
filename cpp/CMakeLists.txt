cmake_minimum_required(VERSION 3.16)
project(MidiEncoder C CXX)

# Set the path to the TensorFlow source code (cloned as a submodule)
set(TENSORFLOW_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../tensorflow)
set(TFLITE_SOURCE_DIR ${TENSORFLOW_SOURCE_DIR}/tensorflow/lite)

# Verify that the TensorFlow directory exists before trying to use it.
# This gives a much clearer error if the path is wrong or the submodule isn't initialized.
if(NOT EXISTS "${TFLITE_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "TensorFlow Lite source not found at '${TENSORFLOW_SOURCE_DIR}'. Please ensure the submodule is cloned and you have opened the project's root folder.")
endif()

# Directly disable TFLite's ability to download dependencies. This forces it
# to use the sources available in the submodule.
set(TFLITE_ENABLE_FETCH_CONTENT OFF CACHE BOOL "Disable TFLite FetchContent")
add_subdirectory(${TFLITE_SOURCE_DIR} tensorflow-lite)

# The TFLite build can sometimes fail to enable the C language for its C-based
# dependencies like XNNPACK. We explicitly enable it here to prevent compilation errors.
set_target_properties(tensorflow-lite PROPERTIES LINKER_LANGUAGE C)

# Add the executable
add_executable(encoder encoder.cpp)

# Set C++ standard for our specific target
set_target_properties(encoder PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

# Link the executable to the TFLite library
target_link_libraries(encoder tensorflow-lite)
