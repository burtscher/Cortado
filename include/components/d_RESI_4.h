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
   Componentized version. Assumes input is float (original data or dequantized from server).
   Encodes the residual between two error bounds. For each value:
     - Quantize to the src and dest eb (in terms of dest)
     - Output the residual between src and dest
   Decoder: Assumes input is dequantized src.
     - Requantize src to dest eb.
     - Apply residual
     - Dequantize result (yielding final decoded float output)
*/


static __device__ inline void d_RESI_4(const int csize, byte in [CS], byte out [CS],
    const int src_eb_e, const int src_thr_e, const int src_offs,
    const int dest_eb_e, const int dest_thr_e, const int dest_offs)
{
  const int size = csize / sizeof(unsigned int);
  const int tid = threadIdx.x;

  const unsigned int* const in_u = (unsigned int*)in;
  unsigned int* const out_u = (unsigned int*)out;

  for (int idx = tid; idx < size; idx += TPB) {
    const int dec = in_u[idx];

    const int src = d_val_client_QUANT_IABS_0_f32(dec, src_eb_e, src_thr_e, src_offs);
    const int src_dequant = d_val_client_iQUANT_IABS_0_f32(src, src_thr_e, src_offs);
    const int src_requant = d_val_client_QUANT_IABS_0_f32(src_dequant, dest_eb_e, dest_thr_e, dest_offs);
    const int dest = d_val_client_QUANT_IABS_0_f32(dec, dest_eb_e, dest_thr_e, dest_offs);
    out_u[idx] = dest - src_requant;
  }
}


static __device__ inline void d_iRESI_4(const int csize, byte in [CS], byte out [CS], byte temp [CS],
    const int dest_eb_e, const int dest_thr_e, const int dest_offs)
{
  const int size = csize / sizeof(unsigned int);
  const int tid = threadIdx.x;

  const unsigned int* const in_u = (unsigned int*)in;  // unquantized src
  const unsigned int* const residual_u = (unsigned int*)temp;  // residual
  unsigned int* const out_u = (unsigned int*)out;  // dequantized dest

  for (int idx = tid; idx < size; idx += TPB) {
    const int src_requant = d_val_client_QUANT_IABS_0_f32(in_u[idx], dest_eb_e, dest_thr_e, dest_offs);
    const int out_quant = src_requant + residual_u[idx];
    out_u[idx] = d_val_client_iQUANT_IABS_0_f32(out_quant, dest_thr_e, dest_offs);
  }
}
