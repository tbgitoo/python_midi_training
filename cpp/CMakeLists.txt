cmake_minimum_required(VERSION 3.16)
project(MidiEncoder CXX)

# Define the path to the pre-compiled TFLite library.
# Update the version number to match the folder you extracted.
set(TFLITE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/vendor/tensorflow-lite-2.16.1)

# Add the executable
add_executable(encoder encoder.cpp)

# Set C++ standard for our specific target
set_target_properties(encoder PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

# Include the TFLite header files
target_include_directories(encoder PRIVATE ${TFLITE_PATH}/include)

# Link the executable to the TFLite library file
target_link_libraries(encoder PRIVATE ${TFLITE_PATH}/lib/tensorflow-lite.lib)

# This command ensures the tensorflow-lite.dll is copied to the same
# directory as your encoder.exe so it can be found at runtime.
add_custom_command(TARGET encoder POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${TFLITE_PATH}/lib/tensorflow-lite.dll"
    $<TARGET_FILE_DIR:encoder>)
