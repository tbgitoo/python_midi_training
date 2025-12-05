cmake_minimum_required(VERSION 3.16)
project(MidiEncoder CXX)

# Set the path to the TensorFlow source code (cloned as a submodule)
set(TENSORFLOW_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../tensorflow)

# Verify that the TensorFlow directory exists before trying to use it.
# This gives a much clearer error if the path is wrong or the submodule isn't initialized.
if(NOT EXISTS "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite/CMakeLists.txt")
    message(FATAL_ERROR "TensorFlow Lite source not found at '${TENSORFLOW_SOURCE_DIR}'. Please ensure the submodule is cloned and you have opened the project's root folder.")
endif()



# Add the TFLite library
add_subdirectory(${TENSORFLOW_SOURCE_DIR}/tensorflow/lite CMAKE_BINARY_DIR)

# Add the executable
add_executable(encoder encoder.cpp)

# Set C++ standard for our specific target
set_target_properties(encoder PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON)

# Link the executable to the TFLite library
target_link_libraries(encoder tensorflow-lite)
