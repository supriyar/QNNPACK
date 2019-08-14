#pragma once
#include <math.h>
#include <array>
#include <string>

#include <qnnpack.h>
#include <qnnpack/log.h>
#include <qnnpack/operator.h>

namespace qnnpack {
struct conv_param_t {
  std::array<uint32_t, 2> kernel_dims; // kernel width, kernel height
  std::array<uint32_t, 2> subsampling_dims; // subsampling width, height
  std::array<uint32_t, 2> dilation; // dilation width, height
  std::array<uint32_t, 4> pad; // input padding top, left, bottom, right
  uint32_t groups;
  size_t input_channels;
  size_t output_channels;
  uint8_t kernel_zero_point;
  float kernel_scale;
  uint8_t output_min;
  uint8_t output_max;

  // The following are derived parameters
  enum qnnp_ukernel_type ukernel_type; // kernel type based on input params
  size_t group_input_channels;
  size_t group_output_channels;

  /**
   * @brief Constructor for initializing the convolution parameters.
   */
  conv_param_t(
      std::array<uint32_t, 2> kernel,
      std::array<uint32_t, 2> subsampling,
      std::array<uint32_t, 2> dil,
      std::array<uint32_t, 4> pd,
      uint32_t grp,
      size_t in_ch,
      size_t out_ch,
      uint8_t kernel_zp,
      float kernel_s,
      uint8_t out_min,
      uint8_t out_max)
      : kernel_dims(kernel),
        subsampling_dims(subsampling),
        dilation(dil),
        pad(pd),
        groups(grp),
        input_channels(in_ch),
        output_channels(out_ch),
        kernel_zero_point(kernel_zp),
        kernel_scale(kernel_s),
        output_min(out_min),
        output_max(out_max)
  {
    uint32_t kernel_width = kernel_dims[0];
    uint32_t kernel_height = kernel_dims[1];

    uint32_t input_padding_top = pad[0];
    uint32_t input_padding_left = pad[1];
    uint32_t input_padding_bottom = pad[2];
    uint32_t input_padding_right = pad[3];

    group_input_channels = input_channels / groups;
    group_output_channels = output_channels / groups;

    if (kernel_width == 0 || kernel_height == 0) {
      qnnp_log_error(
          "failed to create convolution with %" PRIu32 "x%" PRIu32
          " kernel: kernel dimensions must be non-zero",
          kernel_width,
          kernel_height);
      assert("Failed to initialize QNNPACK conv_param_t struct.");
    }

    if (subsampling_dims[0] == 0 || subsampling_dims[1] == 0) {
      qnnp_log_error(
          "failed to create convolution with %" PRIu32 "x%" PRIu32
          " subsampling: "
          "subsampling dimensions must be non-zero",
          subsampling_dims[0],
          subsampling_dims[1]);
      assert("Failed to initialize QNNPACK conv_param_t struct.");
    }

    if (dilation[0] == 0 || dilation[1] == 0) {
      qnnp_log_error(
          "failed to create convolution with %" PRIu32 "x%" PRIu32
          " dilation: "
          "dilation dimensions must be non-zero",
          dilation[0],
          dilation[1]);
      assert("Failed to initialize QNNPACK conv_param_t struct.");
    }

    if (kernel_scale <= 0.0f || !isnormal(kernel_scale)) {
      qnnp_log_error(
          "failed to create convolution with %.7g kernel scale: scale must be"
          "finite and positive",
          kernel_scale);
      assert("Failed to initialize QNNPACK conv_param_t struct.");
    }

    if (subsampling_dims[1] > kernel_height) {
      qnnp_log_info(
          "inefficiency in convolution with %" PRIu32 "x%" PRIu32
          " kernel and %" PRIu32 "x%" PRIu32
          " subsampling: "
          "height subsampling is greater than kernel height; subsampling should"
          " be performed before the convolution",
          kernel_width,
          kernel_height,
          subsampling_dims[0],
          subsampling_dims[1]);
    }

    if (subsampling_dims[0] > kernel_width) {
      qnnp_log_info(
          "inefficiency in convolution with %" PRIu32 "x%" PRIu32
          " kernel and %" PRIu32 "x%" PRIu32
          " subsampling: "
          "width subsampling is greater than kernel width; subsampling should"
          " be performed before the convolution",
          kernel_width,
          kernel_height,
          subsampling_dims[0],
          subsampling_dims[1]);
    }

    if (input_padding_top >= kernel_height) {
      qnnp_log_info(
          "inefficiency in convolution with %" PRIu32 "x%" PRIu32
          " kernel and %" PRIu32 "+%" PRIu32
          " height padding: "
          "input top padding is greater or equal to kernel height",
          kernel_width,
          kernel_height,
          input_padding_top,
          input_padding_bottom);
    }

    if (input_padding_bottom >= kernel_height) {
      qnnp_log_info(
          "inefficiency in convolution with %" PRIu32 "x%" PRIu32
          " kernel and %" PRIu32 "+%" PRIu32
          " height padding: "
          "input bottom padding is greater or equal to kernel height",
          kernel_width,
          kernel_height,
          input_padding_top,
          input_padding_bottom);
    }

    if (input_padding_right >= kernel_width) {
      qnnp_log_info(
          "inefficiency in convolution with %" PRIu32 "x%" PRIu32
          " kernel and %" PRIu32 "+%" PRIu32
          " width padding: "
          "input right padding is greater or equal to kernel width",
          kernel_width,
          kernel_height,
          input_padding_left,
          input_padding_right);
    }

    if (input_padding_left >= kernel_width) {
      qnnp_log_info(
          "inefficiency in convolution with %" PRIu32 "x%" PRIu32
          " kernel and %" PRIu32 "+%" PRIu32
          " width padding: "
          "input left padding is greater or equal to kernel width",
          kernel_width,
          kernel_height,
          input_padding_left,
          input_padding_right);
    }

    const size_t kernel_size = kernel_height * kernel_width;

    ukernel_type = qnnp_ukernel_type_none;
    const bool any_padding = (input_padding_left | input_padding_top
        | input_padding_right | input_padding_bottom) != 0;

    if ((kernel_size == 9 || kernel_size == 25) &&
        group_input_channels == 1 && group_output_channels == 1 && groups > 1) {
      ukernel_type = qnnp_ukernel_type_dwconv;
    } else if (kernel_size == 1 && subsampling_dims[1] == 1 && subsampling_dims[0] == 1 && !any_padding) {
      ukernel_type = group_input_channels >= SIZE_MAX ? qnnp_ukernel_type_xzp_gemm : qnnp_ukernel_type_gemm;
    } else {
      ukernel_type = qnnp_ukernel_type_conv;
    }
  }
};
} // namespace qnnpack
