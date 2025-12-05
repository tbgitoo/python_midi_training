#include <iostream>
#include <vector>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// Define the input and output dimensions
constexpr int kInputRows = 90;
constexpr int kInputCols = 32;
constexpr int kOutputSize = 512;

int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <tflite_model_path>\n", argv[0]);
    return 1;
  }
  const char* model_path = argv[1];

  // 1. Load the model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(model_path);
  if (!model) {
    fprintf(stderr, "Failed to load model from %s\n", model_path);
    return 1;
  }

  // 2. Create an interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    fprintf(stderr, "Failed to construct interpreter\n");
    return 1;
  }

  // 3. Allocate tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    fprintf(stderr, "Failed to allocate tensors\n");
    return 1;
  }

  // 4. Get input and output tensors
  TfLiteTensor* input_tensor = interpreter->input_tensor(0);
  TfLiteTensor* output_tensor = interpreter->output_tensor(0);

  // 5. Create dummy input data (90x32 matrix)
  std::vector<float> input_data(kInputRows * kInputCols);
  for (int i = 0; i < kInputRows * kInputCols; ++i) {
    input_data[i] = static_cast<float>(i) / (kInputRows * kInputCols);
  }

  // 6. Copy input data to input tensor
  memcpy(input_tensor->data.f, input_data.data(),
         input_data.size() * sizeof(float));

  // 7. Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    fprintf(stderr, "Failed to invoke interpreter\n");
    return 1;
  }

  // 8. Extract output data
  const float* output_data = output_tensor->data.f;

  // 9. Print the first few output values
  std::cout << "Output: ";
  for (int i = 0; i < 10; ++i) {
    std::cout << output_data[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}