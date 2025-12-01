#include <optional>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>

#include "attention_dtypes.h"
#include "attention_utils.cuh"

namespace vllm {

// Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005
// can be used to combine partial attention results (in the split-KV case)
template <typename scalar_t, const uint NUM_THREADS>
__global__ void merge_attn_states_kernel(
    scalar_t* output, float* output_lse, const scalar_t* prefix_output,
    const float* prefix_lse, const scalar_t* suffix_output,
    const float* suffix_lse, const uint num_tokens, const uint num_heads,
    const uint head_size, const uint prefix_head_stride,
    const uint output_head_stride) {
  using pack_128b_t = uint4;
  const uint pack_size = 16 / sizeof(scalar_t);
  const uint threads_per_head = head_size / pack_size;

  const uint global_idx = blockIdx.x * NUM_THREADS + threadIdx.x;
  const uint token_head_threads = num_tokens * num_heads * threads_per_head;

  if (global_idx >= token_head_threads) return;

  // global_idx -> token_idx + head_idx + pack_idx
  const uint token_head_idx = global_idx / threads_per_head;
  const uint pack_idx = global_idx % threads_per_head;

  const uint token_idx = token_head_idx / num_heads;
  const uint head_idx = token_head_idx % num_heads;

  const uint pack_offset = pack_idx * pack_size;  // (0~15)*8, etc.
  const uint src_head_offset = token_idx * num_heads * prefix_head_stride +
                               head_idx * prefix_head_stride;
  const uint dst_head_offset = token_idx * num_heads * output_head_stride +
                               head_idx * output_head_stride;
  const scalar_t* prefix_head_ptr = prefix_output + src_head_offset;
  const scalar_t* suffix_head_ptr = suffix_output + src_head_offset;
  scalar_t* output_head_ptr = output + dst_head_offset;

  float p_lse = prefix_lse[head_idx * num_tokens + token_idx];
  float s_lse = suffix_lse[head_idx * num_tokens + token_idx];
  p_lse = std::isinf(p_lse) ? -std::numeric_limits<float>::infinity() : p_lse;
  s_lse = std::isinf(s_lse) ? -std::numeric_limits<float>::infinity() : s_lse;

  const float max_lse = fmaxf(p_lse, s_lse);

  /* In certain edge cases, MLA can produce p_lse = s_lse = -inf;
     continuing the pipeline then yields NaN. Root cause: with chunked prefill
     a batch may be split into two chunks; if a request in that batch has no
     prefix hit, every LSE entry for that request’s position is -inf, and at
     this moment we merge cross-attention at first. For now we simply emit
     prefix_output (expected to be all zeros) and prefix_lse (-inf) to fix
     this problem.
  */
  if (std::isinf(max_lse)) {
    if (pack_offset < head_size) {
      // Pack 128b load
      pack_128b_t p_out_pack = reinterpret_cast<const pack_128b_t*>(
          prefix_head_ptr)[pack_offset / pack_size];

      // Pack 128b storage
      reinterpret_cast<pack_128b_t*>(output_head_ptr)[pack_offset / pack_size] =
          p_out_pack;
    }
    // We only need to write to output_lse once per head.
    if (output_lse != nullptr && pack_idx == 0) {
      output_lse[head_idx * num_tokens + token_idx] = max_lse;
    }
    return;
  }

  // INTERVIEW PROBLEM 5: Merge Attention States
  //
  // Merge prefix and suffix attention outputs using log-sum-exp for stability.
  //
  // Given p_lse/s_lse (log-sum-exp values) and max_lse = max(p_lse, s_lse):
  //   - Compute weights: w_p = exp(p_lse - max_lse), w_s = exp(s_lse - max_lse)
  //   - Normalize: p_scale = w_p/(w_p+w_s), s_scale = w_s/(w_p+w_s)  
  //   - Blend: output = p_scale * prefix_output + s_scale * suffix_output
  //   - If output_lse != nullptr: output_lse = log(w_p + w_s) + max_lse
  //
  // Use 128-bit vectorized loads/stores (pack_128b_t). Convert elements to
  // float for blending via vllm::to_float() / vllm::from_float().
  // Only write output_lse once per head (when pack_idx == 0).
  //
  // Variables available: p_lse, s_lse, max_lse, pack_offset, pack_size, pack_idx,
  // prefix_head_ptr, suffix_head_ptr, output_head_ptr, head_size, head_idx,
  // num_tokens, token_idx, output_lse
  
  // TODO: Implement
}

}  // namespace vllm

// The following macro is used to dispatch the conversion function based on
// the output data type. The FN is a macro that calls a function with
// template<typename scalar_t>.
#define DISPATCH_BY_SCALAR_DTYPE(scalar_dtype, fn)                      \
  {                                                                     \
    if (scalar_dtype == at::ScalarType::Float) {                        \
      fn(float);                                                        \
    } else if (scalar_dtype == at::ScalarType::Half) {                  \
      fn(uint16_t);                                                     \
    } else if (scalar_dtype == at::ScalarType::BFloat16) {              \
      fn(__nv_bfloat16);                                                \
    } else {                                                            \
      TORCH_CHECK(false, "Unsupported data type of O: ", scalar_dtype); \
    }                                                                   \
  }

#define LAUNCH_MERGE_ATTN_STATES(scalar_t, NUM_THREADS)                     \
  {                                                                         \
    vllm::merge_attn_states_kernel<scalar_t, NUM_THREADS>                   \
        <<<grid, block, 0, stream>>>(                                       \
            reinterpret_cast<scalar_t*>(output.data_ptr()), output_lse_ptr, \
            reinterpret_cast<scalar_t*>(prefix_output.data_ptr()),          \
            reinterpret_cast<float*>(prefix_lse.data_ptr()),                \
            reinterpret_cast<scalar_t*>(suffix_output.data_ptr()),          \
            reinterpret_cast<float*>(suffix_lse.data_ptr()), num_tokens,    \
            num_heads, head_size, prefix_head_stride, output_head_stride);  \
  }

/*@brief Merges the attention states from prefix and suffix
 * into the output tensor. NUM_TOKENS: n, NUM_HEADS: h, HEAD_SIZE: d
 *
 * @param output [n,h,d] The output tensor to store the merged attention states.
 * @param output_lse [h,d] Optional tensor to store the log-sum-exp values.
 * @param prefix_output [n,h,d] The prefix attention states.
 * @param prefix_lse [h,n] The log-sum-exp values for the prefix attention
 * states.
 * @param suffix_output [n,h,d] The suffix attention states.
 * @param suffix_lse [h,n] The log-sum-exp values for the suffix attention
 * states.
 */
template <typename scalar_t>
void merge_attn_states_launcher(torch::Tensor& output,
                                std::optional<torch::Tensor> output_lse,
                                const torch::Tensor& prefix_output,
                                const torch::Tensor& prefix_lse,
                                const torch::Tensor& suffix_output,
                                const torch::Tensor& suffix_lse) {
  constexpr uint NUM_THREADS = 128;
  const uint num_tokens = output.size(0);
  const uint num_heads = output.size(1);
  const uint head_size = output.size(2);
  const uint prefix_head_stride = prefix_output.stride(1);
  const uint output_head_stride = output.stride(1);
  const uint pack_size = 16 / sizeof(scalar_t);
  TORCH_CHECK(head_size % pack_size == 0,
              "headsize must be multiple of pack_size:", pack_size);
  float* output_lse_ptr = nullptr;
  if (output_lse.has_value()) {
    output_lse_ptr = output_lse.value().data_ptr<float>();
  }
  // Process one pack elements per thread. for float, the
  // pack_size is 4 for half/bf16, the pack_size is 8.
  const uint threads_per_head = head_size / pack_size;
  const uint total_threads = num_tokens * num_heads * threads_per_head;

  dim3 block(NUM_THREADS);
  dim3 grid((total_threads + NUM_THREADS - 1) / NUM_THREADS);

  const c10::cuda::OptionalCUDAGuard device_guard(prefix_output.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  LAUNCH_MERGE_ATTN_STATES(scalar_t, NUM_THREADS);
}

#define CALL_MERGE_ATTN_STATES_LAUNCHER(scalar_t)                           \
  {                                                                         \
    merge_attn_states_launcher<scalar_t>(output, output_lse, prefix_output, \
                                         prefix_lse, suffix_output,         \
                                         suffix_lse);                       \
  }

void merge_attn_states(torch::Tensor& output,
                       std::optional<torch::Tensor> output_lse,
                       const torch::Tensor& prefix_output,
                       const torch::Tensor& prefix_lse,
                       const torch::Tensor& suffix_output,
                       const torch::Tensor& suffix_lse) {
  DISPATCH_BY_SCALAR_DTYPE(output.dtype(), CALL_MERGE_ATTN_STATES_LAUNCHER);
}
