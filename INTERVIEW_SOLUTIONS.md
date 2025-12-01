# Interview Reference Solutions

> **INTERVIEWER ONLY** - Do not share with candidates

---

## Problem 1: SiLU Activation

```cuda
template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}
```

**Key points:**
- Cast to float for computation
- Compute `x / (1 + exp(-x))` which equals `x * sigmoid(x)`
- Cast back to T

---

## Problem 2: MoE Sum Kernel

```cuda
template <typename scalar_t, int TOPK>
__global__ void moe_sum_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., topk, d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    scalar_t x = 0.0;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      x += VLLM_LDG(&input[token_idx * TOPK * d + k * d + idx]);
    }
    out[token_idx * d + idx] = x;
  }
}
```

**Key points:**
- `blockIdx.x` for token index
- Thread striding: `for (idx = threadIdx.x; idx < d; idx += blockDim.x)`
- Memory layout: `[token][expert][hidden]`
- `#pragma unroll` for the compile-time known TOPK loop

---

## Problem 3: Copy Blocks Kernel

```cuda
template <typename scalar_t>
__global__ void copy_blocks_kernel(int64_t* key_cache_ptrs,
                                   int64_t* value_cache_ptrs,
                                   const int64_t* __restrict__ block_mapping,
                                   const int numel_per_block) {
  const int layer_idx = blockIdx.x;
  const int pair_idx = blockIdx.y;

  scalar_t* key_cache = reinterpret_cast<scalar_t*>(key_cache_ptrs[layer_idx]);
  scalar_t* value_cache =
      reinterpret_cast<scalar_t*>(value_cache_ptrs[layer_idx]);
  int64_t src_block_number = block_mapping[2 * pair_idx];
  int64_t dst_block_number = block_mapping[2 * pair_idx + 1];

  const int64_t src_block_offset = src_block_number * numel_per_block;
  const int64_t dst_block_offset = dst_block_number * numel_per_block;
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    key_cache[dst_offset] = key_cache[src_offset];
  }
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    value_cache[dst_offset] = value_cache[src_offset];
  }
}
```

**Key points:**
- 2D grid: `blockIdx.x` for layer, `blockIdx.y` for pair
- `reinterpret_cast` to convert int64_t pointers to scalar_t pointers
- Block mapping stores pairs as `[src0, dst0, src1, dst1, ...]`
- Thread striding for element copy
- Copy both key and value caches

---

## Problem 4: RoPE Embedding

```cuda
template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_token_rotary_embedding(
    scalar_t* __restrict__ arr, const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr, int rot_offset, int embed_dim) {
  int x_index, y_index;
  scalar_t cos, sin;
  if (IS_NEOX) {
    // GPT-NeoX style rotary embedding.
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = VLLM_LDG(cos_ptr + x_index);
    sin = VLLM_LDG(sin_ptr + x_index);
  } else {
    // GPT-J style rotary embedding.
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = VLLM_LDG(cos_ptr + x_index / 2);
    sin = VLLM_LDG(sin_ptr + x_index / 2);
  }

  const scalar_t x = arr[x_index];
  const scalar_t y = arr[y_index];
  arr[x_index] = x * cos - y * sin;
  arr[y_index] = y * cos + x * sin;
}
```

**Key points:**
- Two different indexing schemes based on `IS_NEOX`
- NeoX: pairs first half with second half `(i, i + embed_dim)`
- GPT-J: pairs adjacent elements `(2i, 2i+1)`
- **Critical:** Read both x and y BEFORE writing either (in-place rotation)
- Rotation formula: standard 2D rotation matrix

---

## Problem 5: Merge Attention States

```cuda
  // After the max_lse computation and edge case handling...
  
  p_lse = p_lse - max_lse;
  s_lse = s_lse - max_lse;
  const float p_se = expf(p_lse);
  const float s_se = expf(s_lse);
  const float out_se = p_se + s_se;
  const float p_scale = p_se / out_se;
  const float s_scale = s_se / out_se;

  if (pack_offset < head_size) {
    // Pack 128b load
    pack_128b_t p_out_pack = reinterpret_cast<const pack_128b_t*>(
        prefix_head_ptr)[pack_offset / pack_size];
    pack_128b_t s_out_pack = reinterpret_cast<const pack_128b_t*>(
        suffix_head_ptr)[pack_offset / pack_size];
    pack_128b_t o_out_pack;

#pragma unroll
    for (uint i = 0; i < pack_size; ++i) {
      // Always use float for FMA to keep high precision.
      const float p_out_f =
          vllm::to_float(reinterpret_cast<const scalar_t*>(&p_out_pack)[i]);
      const float s_out_f =
          vllm::to_float(reinterpret_cast<const scalar_t*>(&s_out_pack)[i]);
      const float o_out_f = p_out_f * p_scale + (s_out_f * s_scale);
      vllm::from_float(reinterpret_cast<scalar_t*>(&o_out_pack)[i], o_out_f);
    }

    // Pack 128b storage
    reinterpret_cast<pack_128b_t*>(output_head_ptr)[pack_offset / pack_size] =
        o_out_pack;
  }
  
  // We only need to write to output_lse once per head.
  if (output_lse != nullptr && pack_idx == 0) {
    float out_lse = logf(out_se) + max_lse;
    output_lse[head_idx * num_tokens + token_idx] = out_lse;
  }
```

**Key points:**
- Log-sum-exp trick: subtract max before exp to prevent overflow
- Weight computation: `exp(lse - max) / (exp(p_lse - max) + exp(s_lse - max))`
- 128-bit vectorized loads/stores with `uint4` / `pack_128b_t`
- Type conversion to float for computation, back to scalar_t for storage
- Final LSE = `log(sum of scaled exps) + max_lse`
- Only one thread per head writes the LSE (`pack_idx == 0`)

---

## Common Mistakes to Watch For

### Problem 1
- Forgetting to cast to float before computation
- Using `exp()` instead of `expf()` (works but less efficient)

### Problem 2
- Using `blockIdx.y` instead of `blockIdx.x` (this is 1D grid)
- Incorrect memory indexing (forgetting TOPK dimension)
- Not using striding (only works for small d)

### Problem 3
- Forgetting `reinterpret_cast`
- Wrong block_mapping indexing (not realizing pairs are consecutive)
- Only copying key_cache, forgetting value_cache

### Problem 4
- Wrong index formulas for NeoX vs GPT-J
- Writing x before reading y (corrupts the rotation)
- Wrong rotation formula signs

### Problem 5
- Not subtracting max_lse before exp (numerical instability)
- Forgetting the vectorized load/store pattern
- Not handling the LSE output correctly
- Missing type conversions

