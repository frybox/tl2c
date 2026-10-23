#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void elementwise_add_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C);
extern "C" __global__ void __launch_bounds__(128, 1) elementwise_add_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* C_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 4096));
  float C_local[8];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    *(float4*)(((float*)A_shared) + ((i * 512) + (((int)threadIdx.x) * 4))) = *(float4*)(A + (((((((int)blockIdx.y) * 32768) + (i * 16384)) + ((((int)threadIdx.x) >> 3) * 1024)) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 4)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    *(float4*)(((float*)B_shared) + ((i_1 * 512) + (((int)threadIdx.x) * 4))) = *(float4*)(B + (((((((int)blockIdx.y) * 32768) + (i_1 * 16384)) + ((((int)threadIdx.x) >> 3) * 1024)) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 4)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    float4 __1;
      float4 v_ = *(float4*)(((float*)A_shared) + ((i_2 * 512) + (((int)threadIdx.x) * 4)));
      float4 v__1 = *(float4*)(((float*)B_shared) + ((i_2 * 512) + (((int)threadIdx.x) * 4)));
      __1.x = (v_.x+v__1.x);
      __1.y = (v_.y+v__1.y);
      __1.z = (v_.z+v__1.z);
      __1.w = (v_.w+v__1.w);
    *(float4*)(C_local + (i_2 * 4)) = __1;
  }
  __syncthreads();
  #pragma unroll
  for (int i_3 = 0; i_3 < 2; ++i_3) {
    *(float4*)(((float*)C_shared) + ((i_3 * 512) + (((int)threadIdx.x) * 4))) = *(float4*)(C_local + (i_3 * 4));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 2; ++i_4) {
    *(float4*)(C + (((((((int)blockIdx.y) * 32768) + (i_4 * 16384)) + ((((int)threadIdx.x) >> 3) * 1024)) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 4))) = *(float4*)(((float*)C_shared) + ((i_4 * 512) + (((int)threadIdx.x) * 4)));
  }
}

