#include <qnnpack.h>
#include <qnnpack/pack.h>
#include <qnnpack_func.h>
#include <cstring>

namespace qnnpack {
PackBMatrix::PackBMatrix(
    size_t input_channels,
    size_t output_channels,
    uint8_t kernel_zero_point,
    float kernel_scale,
    const uint8_t* kernel,
    const int32_t* bias) {
  if (kernel_scale <= 0.0f || !isnormal(kernel_scale)) {
    qnnp_log_error(
        "failed to create fully connected operator with %.7g kernel scale: "
        "scale must be finite and positive",
        kernel_scale);
    assert("QNNPACK Runtime Error.");
  }

  const uint32_t nr = qnnp_params.q8conv.nr;
  const uint32_t kr = qnnp_params.q8conv.kr;

  const uint32_t n_stride = (output_channels + (nr - 1)) & -nr;
  const uint32_t k_stride = (input_channels + (kr - 1)) & -kr;

  input_channels_ = input_channels;
  output_channels_ = output_channels;
  packed_weights_ =
      malloc(n_stride * (k_stride * sizeof(uint8_t) + sizeof(int32_t)));
  if (packed_weights_ == NULL) {
    qnnp_log_error(
        "failed to allocate %zu bytes for packed weights",
        n_stride * (k_stride * sizeof(uint8_t) + sizeof(int32_t)));
    assert("QNNPACK Runtime Error.");
  }
  memset(
      packed_weights_,
      kernel_zero_point,
      n_stride * (k_stride * sizeof(uint8_t) + sizeof(int32_t)));

  // TODO change this
  pack_q8gemm_wrq(
      output_channels,
      input_channels,
      nr,
      nr,
      kr,
      kernel,
      bias,
      packed_weights_);
}
} // namespace qnnpack
