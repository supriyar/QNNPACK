#include <conv_utils.h>
#include <qnnpack.h>
#include <qnnpack/pack.h>
#include <qnnpack_func.h>
#include <cstring>

namespace qnnpack {

PrePackConvWeights::PrePackConvWeights(
    const conv_param_t& conv_p,
    const uint8_t* kernel,
    const int32_t* bias) {
  output_channels_ = conv_p.output_channels;
  enum qnnp_ukernel_type ukernel_type = conv_p.ukernel_type;
  uint32_t kernel_width = conv_p.kernel_dims[0];
  uint32_t kernel_height = conv_p.kernel_dims[1];
  uint32_t groups = conv_p.groups;

  const size_t kernel_size = kernel_height * kernel_width;
  switch (ukernel_type) {
    case qnnp_ukernel_type_dwconv: {
      const uint32_t cr = qnnp_params.q8dw9.cr;
      const uint32_t c_stride = (groups + (cr - 1)) & -cr;
      const size_t packed_weights_size =
          (sizeof(uint8_t) * kernel_size + sizeof(int32_t)) * c_stride;
      packed_weights_ = malloc(packed_weights_size);
      if (packed_weights_ == nullptr) {
        qnnp_log_error(
            "failed to allocate %zu bytes for packed weights",
            packed_weights_size);
        assert("QNNPACK Runtime Error.");
      }

      switch (kernel_size) {
        case 9:
          pack_q8dw_w(
              kernel_height,
              kernel_width,
              groups,
              cr,
              #if !QNNPACK_RUNTIME_QUANTIZATION
              0, /* TODO fix this once kernels are updates input_zero_point,*/
              conv_p.kernel_zero_point,
              #endif
              kernel,
              bias,
              packed_weights_);
          break;
        case 25:
          /* change this later */
          pack_q8dw_w_dilation(
              kernel_height,
              kernel_width,
              groups,
              cr,
              0,
              kernel_height,
              0,
              2,
              kernel,
              bias,
              packed_weights_,
              true);
          pack_q8dw_w_dilation(
              kernel_height,
              kernel_width,
              groups,
              cr,
              0,
              kernel_height,
              2,
              4,
              kernel,
              bias,
              (char*)packed_weights_ +
                  (10 + sizeof(int32_t) / sizeof(uint8_t)) * c_stride,
              false);
          pack_q8dw_w_dilation(
              kernel_height,
              kernel_width,
              groups,
              cr,
              0,
              kernel_height,
              4,
              5,
              kernel,
              bias,
              (char*)packed_weights_ +
                  (20 + sizeof(int32_t) / sizeof(uint8_t)) * c_stride,
              false);
          break;
        default:
          QNNP_UNREACHABLE;
      }
      break;
    }
    case qnnp_ukernel_type_xzp_gemm: {
      const uint32_t nr = qnnp_params.q8conv_xzp.nr;
      const uint32_t kr = qnnp_params.q8conv_xzp.kr;
      const uint32_t sr = qnnp_params.q8conv_xzp.kc;
      const uint32_t n_stride = (conv_p.group_output_channels + (nr - 1)) & -nr;
      const uint32_t k_stride = (conv_p.group_input_channels + (kr - 1)) & -kr;

      const size_t packed_group_weights_size =
          (sizeof(uint8_t) * kernel_size * k_stride + sizeof(int32_t)) *
          n_stride;
      packed_weights_ = malloc(packed_group_weights_size * groups);
      if (packed_weights_ == nullptr) {
        qnnp_log_error(
            "failed to allocate %zu bytes for packed weights",
            packed_group_weights_size * groups);
        assert("QNNPACK Runtime Error.");
      }
      /* The XZP ukernel needs the padding to be 0 */
      memset(packed_weights_, 0, packed_group_weights_size * groups);

      for (uint32_t group = 0; group < groups; group++) {
        pack_swizzle_q8gemm_b(
            conv_p.group_output_channels,
            conv_p.group_input_channels,
            nr,
            kr,
            sr,
            #if !QNNPACK_RUNTIME_QUANTIZATION
            0, /* TODO update this input_zero_point,*/
            conv_p.kernel_zero_point,
            #endif
            kernel +
                group * conv_p.group_output_channels *
                    conv_p.group_input_channels,
            bias + group * conv_p.group_output_channels,
            (void*)((uintptr_t)packed_weights_ + group * packed_group_weights_size));
      }
      break;
    }
    case qnnp_ukernel_type_gemm:
    case qnnp_ukernel_type_conv: {
      const uint32_t nr = qnnp_params.q8conv.nr;
      const uint32_t kr = qnnp_params.q8conv.kr;
      const uint32_t n_stride = (conv_p.group_output_channels + (nr - 1)) & -nr;
      const uint32_t k_stride = (conv_p.group_input_channels + (kr - 1)) & -kr;

      const size_t packed_group_weights_size =
          (sizeof(uint8_t) * kernel_size * k_stride + sizeof(int32_t)) *
          n_stride;
      packed_weights_ = malloc(packed_group_weights_size * groups);
      if (packed_weights_ == nullptr) {
        qnnp_log_error(
            "failed to allocate %zu bytes for packed weights",
            packed_group_weights_size * groups);
        assert("QNNPACK Runtime Error.");
      }
      memset(
          packed_weights_,
          conv_p.kernel_zero_point,
          packed_group_weights_size * groups);

      switch (ukernel_type) {
        case qnnp_ukernel_type_gemm:
          for (uint32_t group = 0; group < groups; group++) {
            pack_q8gemm_w(
                conv_p.group_output_channels,
                conv_p.group_input_channels,
                nr,
                nr,
                kr,
                #if !QNNPACK_RUNTIME_QUANTIZATION
                0, /* TODO Fix this later input_zero_point,*/
                conv_p.kernel_zero_point,
                #endif
                kernel +
                    group * conv_p.group_output_channels *
                        conv_p.group_input_channels,
                bias + group * conv_p.group_output_channels,
                (void*)((uintptr_t)packed_weights_ + group * packed_group_weights_size));
          }
          break;
        case qnnp_ukernel_type_conv:
          for (uint32_t group = 0; group < groups; group++) {
            pack_q8conv_w(
                conv_p.group_output_channels,
                kernel_size,
                conv_p.group_input_channels,
                nr,
                kr,
                #if !QNNPACK_RUNTIME_QUANTIZATION
                0, /* TODO Fix this later input_zero_point,*/
                conv_p.kernel_zero_point,
                #endif
                kernel +
                    group * conv_p.group_output_channels * kernel_size *
                        conv_p.group_input_channels,
                bias + group * conv_p.group_output_channels,
                (void*)((uintptr_t)packed_weights_ + group * packed_group_weights_size));
          }
          break;
        default:
          QNNP_UNREACHABLE;
      }
      break;
    }
    default:
      QNNP_UNREACHABLE;
  }
} // namespace qnnpack
} // namespace qnnpack
