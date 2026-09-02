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


#include "include/d_compression.h"


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
static __global__ __launch_bounds__(TPB, 4)
#else
static __global__ __launch_bounds__(TPB, 2)
#endif
void d_server_encode_kernel(
    const byte* const __restrict__ input, const long long insize, byte* const __restrict__ output, long long* const __restrict__ outsize, long long* const __restrict__ fullcarry, 
    const int eb_e, const int thr_e, const int eb_offs)
{
  // allocate shared memory buffer
  __shared__ long long chunk [2 * (CS / sizeof(long long)) + 309];  // + 132 bytes (block sum reduction) + 2340 bytes (recursive bitmaps)

  // split into 3 shared memory buffers
  byte* in = (byte*)&chunk[0 * (CS / sizeof(long long))];
  byte* out = (byte*)&chunk[1 * (CS / sizeof(long long))];
  byte* const temp = (byte*)&chunk[2 * (CS / sizeof(long long))];

  // initialize
  const int tid = threadIdx.x;
  const long long last = 2 * (CS / sizeof(long long)) + 1;
  const long long chunks = (insize + CS - 1) / CS;  // round up
  long long* const head_out = (long long*)output;
  unsigned short* const size_out = (unsigned short*)&head_out[2];
  byte* const data_out = (byte*)&size_out[chunks];

  // loop over chunks
  do {
    // assign work dynamically
    if (tid == 0) chunk[last] = atomicAdd(&g_chunk_counter, 1LL);
    __syncthreads();  // chunk[last] produced, chunk consumed

    // terminate if done
    const long long chunkID = chunk[last];
    const long long base = chunkID * CS;
    if (base >= insize) break;

    // load chunk
    const int osize = (int)min((long long)CS, insize - base);
    long long* const input_l = (long long*)&input[base];
    long long* const out_l = (long long*)out;
    for (int i = tid; i < osize / 8; i += TPB) {
      out_l[i] = input_l[i];
    }
    const int extra = osize % 8;
    if (tid < extra) out[(long long)osize - (long long)extra + (long long)tid] = input[base + (long long)osize - (long long)extra + (long long)tid];

    // encode chunk
    __syncthreads();  // chunk produced, chunk[last] consumed
    int csize = osize;
    bool good = true;
    {
      byte* tmp;
      tmp = in; in = out; out = tmp;
      d_SQ_IABS_4(csize, in, out, eb_e, thr_e, eb_offs);
      __syncthreads();
      tmp = in; in = out; out = tmp;
      d_DIFFMS_4(csize, in, out, temp);
      __syncthreads();
      tmp = in; in = out; out = tmp;
      d_BIT_4(csize, in, out, temp);
      __syncthreads();
      tmp = in; in = out; out = tmp;
      good = d_RZE_1(csize, in, out, temp);
      __syncthreads();
    }

    // handle carry
    if (!good || (csize >= osize)) csize = osize;
    propagate_carry(csize, chunkID, fullcarry, (long long*)temp);

    // reload chunk if incompressible
    if (tid == 0) size_out[chunkID] = csize;
    if (csize == osize) {
      // store original data
      long long* const out_l = (long long*)out;
      for (long long i = tid; i < osize / 8; i += TPB) {
        out_l[i] = input_l[i];
      }
      const int extra = osize % 8;
      if (tid < extra) out[(long long)osize - (long long)extra + (long long)tid] = input[base + (long long)osize - (long long)extra + (long long)tid];
    }
    __syncthreads();  // "out" done, temp produced

    // store chunk
    const long long offs = (chunkID == 0) ? 0 : *((long long*)temp);
    s2g(&data_out[offs], out, csize);

    // finalize if last chunk
    if ((tid == 0) && (base + CS >= insize)) {
      // output header
      head_out[0] = insize;
      int* const head_out_eb = (int*)&head_out[1];
      head_out_eb[0] = eb_e;
      // compute compressed size
      *outsize = &data_out[fullcarry[chunkID]] - output;
    }
  } while (true);
}


static inline void d_server_encode(const byte* const d_input, const long long insize, byte* const d_output, long long* const d_outsize,
    const float eb, const int chunks, const int blocks)
{
  const int e = 8;  // exponent bits
  const int m = 23;  // mantissa bits

  const float server_errorbound = eb / 2;
  if (server_errorbound < std::numeric_limits<float>::min()) {fprintf(stderr, "ERROR: server-stored errorbound is too small\n"); throw std::runtime_error("LC error");}  // minimum positive normalized value

  const int eb_e = ((*((int*)&server_errorbound) >> m) & ((1 << e) - 1));  // extract biased exponent
  const int thr_e = eb_e + (m + 1);  // biased exponent of threshold
  const int eb_offs = (thr_e << m) - (1 << m);  // offset for lossless encoding
  if (thr_e >= (1 << e) - 1) {fprintf(stderr, "ERROR: server-stored errorbound is too large\n"); throw std::runtime_error("LC error");}

  long long* d_fullcarry;
  cudaMalloc((void **)&d_fullcarry, chunks * sizeof(long long));
  d_reset<<<1, 1>>>();
  cudaMemsetAsync(d_fullcarry, 0, chunks * sizeof(long long));
  d_server_encode_kernel<<<blocks, TPB>>>(d_input, insize, d_output, d_outsize, d_fullcarry, eb_e, thr_e, eb_offs);
  cudaFree(d_fullcarry);
}


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
static __global__ __launch_bounds__(TPB, 4)
#else
static __global__ __launch_bounds__(TPB, 2)
#endif
void d_server_decode_kernel(const byte* const __restrict__ input, byte* const __restrict__ output, long long* const __restrict__ g_outsize)
{
  // allocate shared memory buffer
  __shared__ long long chunk [2 * (CS / sizeof(long long)) + 309];  // + 132 bytes (block sum reduction) + 2340 bytes (recursive bitmaps)
  const int last = 2 * (CS / sizeof(long long)) + 1;

  // create the 3 shared memory buffers
  byte* in = (byte*)&chunk[0 * (CS / sizeof(long long))];
  byte* out = (byte*)&chunk[1 * (CS / sizeof(long long))];
  byte* temp = (byte*)&chunk[2 * (CS / sizeof(long long))];

  const int m = 23;  // mantissa bits

  // input header
  long long* const head_in = (long long*)input;
  int* const head_in_eb = (int*)&head_in[1];
  const long long outsize = head_in[0];
  const int eb_e = head_in_eb[0];
  const int thr_e = eb_e + (m + 1);  // biased exponent of threshold
  const int eb_offs = (thr_e << m) - (1 << m);  // offset for lossless encoding

  // initialize
  const long long chunks = (outsize + CS - 1) / CS;  // round up
  unsigned short* const size_in = (unsigned short*)&head_in[2];
  byte* const data_in = (byte*)&size_in[chunks];

  // loop over chunks
  const int tid = threadIdx.x;
  long long prevChunkID = 0;
  long long prevOffset = 0;
  do {
    // assign work dynamically
    if (tid == 0) chunk[last] = atomicAdd(&g_chunk_counter, 1LL);
    __syncthreads();  // chunk[last] produced, chunk consumed

    // terminate if done
    const long long chunkID = chunk[last];
    const long long base = chunkID * CS;
    if (base >= outsize) break;

    // compute sum of all prior csizes (start where left off in previous iteration)
    long long sum = 0;
    for (long long i = prevChunkID + tid; i < chunkID; i += TPB) {
      sum += (long long)size_in[i];
    }
    int csize = (int)size_in[chunkID];
    const long long offs = prevOffset + block_sum_reduction(sum, &chunk[last + 1]);
    prevChunkID = chunkID;
    prevOffset = offs;

    // load chunk
    g2s(in, &data_in[offs], csize, out);
    byte* tmp = in; in = out; out = tmp;
    __syncthreads();  // chunk produced, chunk[last] consumed

    // decode
    const int osize = (int)min((long long)CS, outsize - base);
    if (csize < osize) {
      byte* tmp;
      tmp = in; in = out; out = tmp;
      d_iRZE_1(csize, in, out, temp);
      __syncthreads();
      tmp = in; in = out; out = tmp;
      d_iBIT_4(csize, in, out, temp);
      __syncthreads();
      tmp = in; in = out; out = tmp;
      d_iDIFFMS_4(csize, in, out, temp);
      __syncthreads();
      tmp = in; in = out; out = tmp;
      d_iSQ_IABS_4(csize, in, out, thr_e, eb_offs);
      __syncthreads();
    }

    if (csize != osize) {printf("ERROR: csize %d doesn't match osize %d in chunk %lld\n\n", csize, osize, chunkID); __trap();}
    long long* const output_l = (long long*)&output[base];
    long long* const out_l = (long long*)out;
    for (int i = tid; i < osize / 8; i += TPB) {
      output_l[i] = out_l[i];
    }
    const int extra = osize % 8;
    if (tid < extra) output[base + osize - extra + tid] = out[osize - extra + tid];
  } while (true);

  if ((blockIdx.x == 0) && (tid == 0)) {
    *g_outsize = outsize;
  }
}


static inline void d_server_decode(const byte* const d_input, byte* const d_output, long long* const d_outsize, const int blocks)
{
  d_reset<<<1, 1>>>();
  d_server_decode_kernel<<<blocks, TPB>>>(d_input, d_output, d_outsize);
}

