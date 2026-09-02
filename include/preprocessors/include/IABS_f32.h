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


#ifndef GPU_IABS_f32
#define GPU_IABS_f32

/*
   Single-value quantizers
*/


static __device__ inline int d_val_server_iQUANT_IABS_0_f32(const int enc, const int thr_e, const int offs)
{
  const int e = 8;  // exponent bits
  const int m = 23;  // mantissa bits
  int dec = 0;  // default value is 0
  if (enc != 0) {
    const int abs = (enc < 0) ? -enc : enc;  // absolute value (TC->magnitude)
    if (abs >= (1 << m)) {  // above threshold
      dec = abs + offs;  // decode losslessly
    } else {  // non-zero lossy case
      const int shift = __builtin_clz(abs) - (31 - m);  // compute shift amount
      dec = abs << shift;  // shift to normalized position
      dec &= (1 << m) - 1;  // remove implied 1
      dec |= (thr_e - shift) << m;  // insert biased exponent
    }
    dec |= enc & ((unsigned int)1 << (e + m));  // insert sign bit
  }
  return dec;
}


static __device__ inline int d_val_client_QUANT_IABS_0_f32(const int val, const int eb_e, const int thr_e, const int offs)
{
  const int e = 8;  // exponent bits
  const int m = 23;  // mantissa bits
  const int abs = val & (((unsigned int)1 << (e + m)) - 1);  // compute absolute value
  const int val_e = abs >> m;  // extract exponent
  int enc = 0;  // default value is 0
  if (val_e >= thr_e) {  // at or above threshold
    enc = abs - offs;  // lossless encoding
  } else if (val_e >= eb_e) {  // lossy encoding
    int mant = val & ((1 << m) - 1);  // extract mantissa
    const int shift = thr_e - val_e;  // bias cancels out
    mant |= 1 << m;  // insert implicit 1
    mant += 1 << (shift - 1);  // round to nearest, ties round away from zero
    enc = mant >> shift;  // shift out unnecessary bits
  }
  if (val >> (e + m)) enc = -enc;  // reintroduce sign
  return enc;
}


static __device__ inline int d_val_client_iQUANT_IABS_0_f32(const int val, const int thr_e, const int offs)
{
  // NOTE: identical to server iQUANT
  const int e = 8;  // exponent bits
  const int m = 23;  // mantissa bits
  const int enc = val;
  int dec = 0;  // default value is 0
  if (enc != 0) {
    const int abs = (enc < 0) ? -enc : enc;  // absolute value (TC->magnitude)
    if (abs >= (1 << m)) {  // above threshold
      dec = abs + offs;  // decode losslessly
    } else {  // non-zero lossy case
      const int shift = __builtin_clz(abs) - (31 - m);  // compute shift amount
      dec = abs << shift;  // shift to normalized position
      dec &= (1 << m) - 1;  // remove implied 1
      dec |= (thr_e - shift) << m;  // insert biased exponent
    }
    dec |= enc & ((unsigned int)1 << (e + m));  // insert sign bit
  }
  return dec;
}


#endif
