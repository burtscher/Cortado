/*
This file is part of Cortado, a guaranteed error-bounded progressive compressor for floating-point data on GPU.

BSD 3-Clause License

Copyright (c) 2026, Brandon Alexander Burtchell and Martin Burtscher
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

URL: The latest version of this code is available at https://github.com/burtscher/Cortado.

Publication: This work is described in detail in the following paper.
Brandon Alexander Burtchell and Martin Burtscher. "Cortado: Progressive Retrieval of Lossily Compressed Data with Guaranteed Error Bounds." Proceedings of the IEEE International Conference on Cluster Computing. September 2026.

Sponsor: This work has been supported by the U.S. National Science Foundation (NSF) under Award CCF-2403380, by the Department of Energy (DOE), Office of Science, Advanced Scientific Computing Research (ASCR) under Award DE-SC0022223, and by an equipment donation from NVIDIA Corporation.
*/


#include "include/preprocessors/include/IABS_f32.h"

/*
   Encodes the residual between two error bounds from a tighter server-stored version. For each value:
     - Decode the server-stored bin into a float
     - Requantize at both the src and dest eb
     - Output the residual between src and dest
*/


static __global__ void d_residual_QUANT_IABS_0_f32_kernel(
    const long long len, unsigned int* const __restrict__ data_u,
    const int server_thr_e, const int server_offs,
    const int src_eb_e, const int src_thr_e, const int src_offs,
    const int dest_eb_e, const int dest_thr_e, const int dest_offs)
{
  const long long idx = threadIdx.x + (long long)blockIdx.x * TPB;
  if (idx < len) {
    const int dec = d_val_server_iQUANT_IABS_0_f32(data_u[idx], server_thr_e, server_offs);
    const int src = d_val_client_QUANT_IABS_0_f32(dec, src_eb_e, src_thr_e, src_offs);
    const int src_dequant = d_val_client_iQUANT_IABS_0_f32(src, src_thr_e, src_offs);
    const int src_requant = d_val_client_QUANT_IABS_0_f32(src_dequant, dest_eb_e, dest_thr_e, dest_offs);
    const int dest = d_val_client_QUANT_IABS_0_f32(dec, dest_eb_e, dest_thr_e, dest_offs);
    data_u[idx] = dest - src_requant;
  }
}


static inline void d_residual_QUANT_IABS_0_f32(long long& size, byte*& data, const float src_eb, const float dest_eb)
{
  const int e = 8;  // exponent bits
  const int m = 23;  // mantissa bits
  if (size % sizeof(float) != 0) {fprintf(stderr, "residual_QUANT_IABS_0_f32: ERROR: size of input must be a multiple of %ld bytes\n", sizeof(float)); throw std::runtime_error("LC error");}

  size -= sizeof(float);
  const long long len = size / sizeof(float);
  unsigned int* const data_u = (unsigned int*)data;

  // read stored server eb from file
  int server_eb_e;
  cudaMemcpy(&server_eb_e, &data_u[len], sizeof(int), cudaMemcpyDeviceToHost);  // use exponent of adjusted error bound (ignore passed parameter)
  const int server_thr_e = server_eb_e + (m + 1);  // biased exponent of threshold
  const int server_offs = (server_thr_e << m) - (1 << m);  // offset for lossless encoding
  if (server_thr_e >= (1 << e) - 1) {fprintf(stderr, "residual_QUANT_IABS_0_f32: ERROR: stored server eb is too large\n"); throw std::runtime_error("LC error");}

  if (src_eb < std::numeric_limits<float>::min()) {fprintf(stderr, "residual_QUANT_IABS_0_f32: ERROR: src eb is too small\n"); throw std::runtime_error("LC error");}  // minimum positive normalized value
  const int src_eb_e = ((*((int*)&src_eb) >> m) & ((1 << e) - 1));  // extract biased exponent
  const int src_thr_e = src_eb_e + (m + 1);  // biased exponent of threshold
  const int src_offs = (src_thr_e << m) - (1 << m);  // offset for lossless encoding
  if (src_thr_e >= (1 << e) - 1) {fprintf(stderr, "residual_QUANT_IABS_0_f32: ERROR: src eb is too large\n"); throw std::runtime_error("LC error");}

  if (dest_eb < std::numeric_limits<float>::min()) {fprintf(stderr, "residual_QUANT_IABS_0_f32: ERROR: dest eb is too small\n"); throw std::runtime_error("LC error");}  // minimum positive normalized value
  const int dest_eb_e = ((*((int*)&dest_eb) >> m) & ((1 << e) - 1));  // extract biased exponent
  const int dest_thr_e = dest_eb_e + (m + 1);  // biased exponent of threshold
  const int dest_offs = (dest_thr_e << m) - (1 << m);  // offset for lossless encoding
  if (dest_thr_e >= (1 << e) - 1) {fprintf(stderr, "residual_QUANT_IABS_0_f32: ERROR: dest eb is too large\n"); throw std::runtime_error("LC error");}

  cudaMemcpyAsync(&data_u[len], &dest_eb_e, sizeof(int), cudaMemcpyHostToDevice);
  d_residual_QUANT_IABS_0_f32_kernel<<<(len + TPB - 1) / TPB, TPB>>>(len, data_u, server_thr_e, server_offs, src_eb_e, src_thr_e, src_offs, dest_eb_e, dest_thr_e, dest_offs);

  size += sizeof(float);
}
