# CUDA Engineer Interview Guide

## Overview

This vLLM fork has been modified for interviewing CUDA engineers. Five kernel implementations have been removed and replaced with succinct problem descriptions. Candidates are expected to reimplement these functions based on the comments and documentation.

**Total Interview Time:** 1 hour  
**Recommended Problem Time:** 35-45 minutes (leaving buffer for setup, questions, and discussion)

---

## Problems Summary

| # | Problem | File | Difficulty | Time | CUDA Concepts |
|---|---------|------|------------|------|---------------|
| 1 | SiLU Activation | `csrc/activation_kernels.cu` | Easy | ~1 min | Device functions, basic math |
| 2 | MoE Sum | `csrc/moe/moe_align_sum_kernels.cu` | Easy-Medium | ~5 min | Grid/block indexing, strided loops |
| 3 | Copy Blocks | `csrc/cache_kernels.cu` | Medium | ~8 min | 2D grids, pointer casting, memory copy |
| 4 | RoPE Embedding | `csrc/pos_encoding_kernels.cu` | Medium-Hard | ~12 min | Complex indexing, trig math, templates |
| 5 | Merge Attention States | `csrc/attention/merge_attn_states.cu` | Hard | ~15 min | Numerical stability, vectorization |

**Total Expected Time for All Problems:** ~41 minutes

---

## Problem Details

### Problem 1: SiLU Activation (~1 minute)
**Location:** `csrc/activation_kernels.cu` → `silu_kernel()`

**What to look for:**
- Knows sigmoid formula
- Proper float casting and `expf()`

---

### Problem 2: MoE Sum Kernel (~5 minutes)
**Location:** `csrc/moe/moe_align_sum_kernels.cu` → `moe_sum_kernel()`

**What to look for:**
- Uses `blockIdx.x` for token, threads stride through `d`
- Correct 3D indexing into `[tokens, TOPK, d]` layout
- Accumulates across TOPK dimension

---

### Problem 3: Copy Blocks Kernel (~8 minutes)
**Location:** `csrc/cache_kernels.cu` → `copy_blocks_kernel()`

**What to look for:**
- Extracts layer/pair indices from 2D grid
- Casts int64_t pointers to scalar_t*
- Correctly interprets block_mapping as [src,dst] pairs
- Copies both key and value caches

---

### Problem 4: RoPE Embedding (~12 minutes)
**Location:** `csrc/pos_encoding_kernels.cu` → `apply_token_rotary_embedding()`

**What to look for:**
- Different indexing for NeoX vs GPT-J styles
- Correct rotation: `x' = x*cos - y*sin`, `y' = y*cos + x*sin`
- Reads both values before writing (in-place safety)

---

### Problem 5: Merge Attention States (~15 minutes)
**Location:** `csrc/attention/merge_attn_states.cu` → Inside `merge_attn_states_kernel()`

**What to look for:**
- Subtracts max_lse before exp (numerical stability)
- Correct weight normalization
- 128-bit vectorized access pattern
- Type conversions for blending
- Correct combined LSE formula

---

## Interview Flow Options

### Standard Path (All candidates)
1. **Warm-up (Problem 1):** ~1-2 min
2. **Problem 2:** ~5-7 min
3. **Problem 3:** ~8-10 min
4. **Problem 4 or 5:** ~12-15 min
5. Discussion: remaining time

### For Strong Candidates
- Can skip Problem 1 after brief explanation
- Move faster through Problems 2-3
- Focus on Problems 4 and 5
- Ask optimization questions about their solutions

### For Junior/Learning Candidates
- Spend more time on Problems 1-3
- Use Problem 4 as a stretch goal
- Skip Problem 5 or discuss it conceptually
- Provide hints more freely

### Alternative Problem Sets

**Path A: Systems/Memory Focus**
- Problems 1, 2, 3, and briefly discuss 4

**Path B: Math/Algorithms Focus**  
- Problems 1, 2, 4, 5

**Path C: Quick Assessment (30 min)**
- Problems 1, 2, and choice of 3 or 4

---

## Setup Instructions

### For the Candidate
1. Clone this repository
2. Open in their preferred editor/IDE
3. Navigate to the problem files
4. Each problem has a detailed comment block explaining requirements

### Development Environment
No compilation required during the interview - this is a code writing exercise. If candidates want to test:
```bash
# Requires CUDA toolkit and PyTorch
pip install -e .
```

---

## Evaluation Rubric

### Technical Correctness (50%)
- Algorithm is correct
- Edge cases handled appropriately
- Memory access patterns are valid

### CUDA Knowledge (30%)
- Proper use of thread/block indexing
- Understanding of memory hierarchy
- Awareness of common patterns (striding, coalescing)

### Code Quality (20%)
- Clean, readable code
- Appropriate comments
- Follows existing code style

---

## Discussion Questions

After coding, consider asking:

### Performance
- "How would you optimize this kernel further?"
- "What's the memory access pattern? Is it coalesced?"
- "Would shared memory help here?"

### Understanding
- "What happens if the input size isn't a multiple of the block size?"
- "Why do we use the log-sum-exp trick in Problem 5?"
- "What's the difference between `__ldg` and regular loads?"

### Architecture
- "How would this change for AMD GPUs (HIP)?"
- "What occupancy considerations are there?"

---

## Reference Solutions

The original implementations can be found by checking out the `main` branch:
```bash
git checkout main
git diff HEAD~1 -- csrc/  # View changes
```

Or refer to the vLLM upstream repository: https://github.com/vllm-project/vllm

---

## Notes for Interviewer

1. **Timing is flexible** - The estimates assume a strong candidate. Adjust based on how the candidate is doing.

2. **Hints are okay** - The goal is to see how they think, not to trick them. Provide hints if stuck.

3. **Partial solutions count** - A candidate who gets 80% of a hard problem is doing well.

4. **Discussion is valuable** - If they finish early, discuss optimizations, alternatives, or edge cases.

5. **Watch for red flags:**
   - Not understanding thread/block model at all
   - Unable to reason about memory layouts
   - Copy-pasting without understanding

6. **Green flags:**
   - Asks clarifying questions
   - Considers edge cases unprompted
   - Discusses performance implications
   - Recognizes patterns from other CUDA code

---

## Customization

Feel free to modify problem difficulty by:
- **Making easier:** Provide more of the boilerplate, reduce requirements
- **Making harder:** Ask for optimizations, remove hints from comments
- **Adding constraints:** "Assume head_size is always 128", etc.

