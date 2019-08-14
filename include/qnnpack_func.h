#pragma once
#include <conv_utils.h>

namespace qnnpack {
class PrePackConvWeights {
 public:
  PrePackConvWeights() = delete; // no default constructor

  PrePackConvWeights(const conv_param_t& conv_param, const uint8_t* kernel, const int32_t* bias);

  void* get_packed_weights()
  {
    return packed_weights_;
  }

  int64_t get_output_channels()
  {
    return output_channels_;
  }

  ~PrePackConvWeights()
  {
    if (packed_weights_ != nullptr) {
      free(packed_weights_);
    }
  }

 private:
  void* packed_weights_ = nullptr;
  int64_t output_channels_;
};

class PackBMatrix {
 public:
  PackBMatrix() = delete; // no default constructor

  PackBMatrix(
      size_t input_channels,
      size_t output_channels,
      uint8_t kernel_zero_point,
      float kernel_scale,
      const uint8_t* kernel,
      const int32_t* bias);

  void* get_packed_weights()
  {
    return packed_weights_;
  }

  size_t get_input_channels()
  {
    return input_channels_;
  }

  size_t get_output_channels()
  {
    return output_channels_;
  }

  ~PackBMatrix()
  {
    if (packed_weights_ != nullptr) {
      free(packed_weights_);
    }
  }

 private:
  void* packed_weights_ = nullptr;
  size_t input_channels_;
  size_t output_channels_;
};

enum qnnp_status qnnpackLinear(
    size_t batch_size,
    size_t input_channels,
    size_t output_channels,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t kernel_zero_point,
    float kernel_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    const uint8_t* input,
    size_t input_stride,
    void* packed_weights,
    uint8_t* output,
    size_t output_stride,
    pthreadpool_t threadpool);

enum qnnp_status qnnpackConv(
    const conv_param_t& conv_p,
    void* packed_weights,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    float input_scale,
    uint8_t input_zero_point,
    const uint8_t* input,
    float output_scale,
    uint8_t output_zero_point,
    uint8_t* output,
    pthreadpool_t threadpool);

} // namespace qnnpack
