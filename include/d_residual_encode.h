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
static __global__ __launch_bounds__(TPB, 3)
#else
static __global__ __launch_bounds__(TPB, 2)
#endif
void d_residual_encode_kernel(
    const byte* const __restrict__ input, byte* const __restrict__ output, long long* const __restrict__ outsize, long long* const __restrict__ fullcarry,
    const int src_eb_e, const int src_thr_e, const int src_eb_offs,
    const int dest_eb_e, const int dest_thr_e, const int dest_eb_offs)
{
  // allocate shared memory buffer
  __shared__ long long chunk [3 * (CS / sizeof(long long))];
  const int last = 3 * (CS / sizeof(long long)) - 2 - WS;

  // create the 3 shared memory buffers
  byte* in = (byte*)&chunk[0 * (CS / sizeof(long long))];
  byte* out = (byte*)&chunk[1 * (CS / sizeof(long long))];
  byte* temp = (byte*)&chunk[2 * (CS / sizeof(long long))];

  const int m = 23;  // mantissa bits

  // input header
  long long* const head_in = (long long*)input;
  int* const head_in_eb = (int*)&head_in[1];
  const long long orig_size = head_in[0];  // original size of uncompressed input
  const int server_eb_e = head_in_eb[0];
  const int server_thr_e = server_eb_e + (m + 1);  // biased exponent of threshold
  const int server_eb_offs = (server_thr_e << m) - (1 << m);  // offset for lossless encoding

  // initialize input
  const long long chunks = (orig_size + CS - 1) / CS;  // round up
  unsigned short* const size_in = (unsigned short*)&head_in[2];
  byte* const data_in = (byte*)&size_in[chunks];

  // initialize output
  long long* const head_out = (long long*)output;
  unsigned short* const size_out = (unsigned short*)&head_out[2];
  byte* const data_out = (byte*)&size_out[chunks];

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
    if (base >= orig_size) break;

    // compute sum of all prior csizes (start where left off in previous iteration)
    long long sum = 0;
    for (long long i = prevChunkID + tid; i < chunkID; i += TPB) {
      sum += (long long)size_in[i];
    }
    int csize = (int)size_in[chunkID];
    const long long in_offs = prevOffset + block_sum_reduction(sum, &chunk[last + 1]);
    prevChunkID = chunkID;
    prevOffset = in_offs;

    // load chunk
    g2s(in, &data_in[in_offs], csize, out);
    byte* tmp = in; in = out; out = tmp;
    __syncthreads();  // chunk produced, chunk[last] consumed

    // decode
    const int osize = (int)min((long long)CS, orig_size - base);
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
      d_iSQ_IABS_4(csize, in, out, server_thr_e, server_eb_offs);
      __syncthreads();
    }
    if (csize != osize) {printf("ERROR: csize %d doesn't match osize %d in chunk %lld\n\n", csize, osize, chunkID); __trap();}

    {
      // always send residual even if incompressable
      byte* tmp = in; in = out; out = tmp;
      d_RESI_4(csize, in, out, src_eb_e, src_thr_e, src_eb_offs, dest_eb_e, dest_thr_e, dest_eb_offs);
      __syncthreads();
    }

    long long hold;
    {
      long long* const out_l = (long long*)out;
      if (tid < 309) {
        hold = out_l[tid];  // backup values before overwriting
      }
      __syncthreads();
    }

    // encode chunk
    bool good = true;
    {
      byte* tmp;
      tmp = in; in = out; out = tmp;
      d_DIFFMS_4(csize, in, out, temp);
      __syncthreads();
      tmp = in; in = temp; temp = tmp;  // move input bin residuals (from RESI) to temp
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
      // restore backup from hold/temp
      long long* const out_l = (long long*)out;
      long long* const temp_l = (long long*)temp;
      for (long long i = tid; i < (osize + 7) / 8; i += TPB) {
        long long val;
        if (i < 309) {
          val = hold;
        } else {
          val = temp_l[i];
        }
        out_l[i] = val;
      }
    }
    __syncthreads();  // "out" done, temp produced

    // store chunk
    const long long out_offs = (chunkID == 0) ? 0 : *((long long*)temp);
    s2g(&data_out[out_offs], out, csize);

    // finalize if last chunk
    if ((tid == 0) && (base + CS >= orig_size)) {
      // output header
      head_out[0] = orig_size;
      int* const head_out_eb = (int*)&head_out[1];
      head_out_eb[0] = dest_eb_e;
      // compute compressed size
      *outsize = &data_out[fullcarry[chunkID]] - output;
    }
  } while (true);
}


static inline void d_residual_encode(const byte* const d_input, byte* const d_output, long long* const d_outsize,
    const float src_eb, const float dest_eb, const int chunks, const int blocks)
{
  static_assert(TPB > 309);

  const int e = 8;  // exponent bits
  const int m = 23;  // mantissa bits

  if (src_eb < std::numeric_limits<float>::min()) {fprintf(stderr, "ERROR: client source errorbound is too small\n"); throw std::runtime_error("LC error");}  // minimum positive normalized value
  const int src_eb_e = ((*((int*)&src_eb) >> m) & ((1 << e) - 1));  // extract biased exponent
  const int src_thr_e = src_eb_e + (m + 1);  // biased exponent of threshold
  const int src_offs = (src_thr_e << m) - (1 << m);  // offset for lossless encoding
  if (src_thr_e >= (1 << e) - 1) {fprintf(stderr, "ERROR: client source errorbound is too large\n"); throw std::runtime_error("LC error");}

  if (dest_eb < std::numeric_limits<float>::min()) {fprintf(stderr, "ERROR: client destination errorbound is too small\n"); throw std::runtime_error("LC error");}  // minimum positive normalized value
  const int dest_eb_e = ((*((int*)&dest_eb) >> m) & ((1 << e) - 1));  // extract biased exponent
  const int dest_thr_e = dest_eb_e + (m + 1);  // biased exponent of threshold
  const int dest_offs = (dest_thr_e << m) - (1 << m);  // offset for lossless encoding
  if (dest_thr_e >= (1 << e) - 1) {fprintf(stderr, "ERROR: client destination errorbound is too large\n"); throw std::runtime_error("LC error");}

  long long* d_fullcarry;
  cudaMalloc((void **)&d_fullcarry, chunks * sizeof(long long));
  d_reset<<<1, 1>>>();
  cudaMemsetAsync(d_fullcarry, 0, chunks * sizeof(long long));
  d_residual_encode_kernel<<<blocks, TPB>>>(d_input, d_output, d_outsize, d_fullcarry, src_eb_e, src_thr_e, src_offs, dest_eb_e, dest_thr_e, dest_offs);
  cudaFree(d_fullcarry);
}


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
static __global__ __launch_bounds__(TPB, 3)
#else
static __global__ __launch_bounds__(TPB, 2)
#endif
void d_residual_apply_kernel(const byte* const __restrict__ input_src, const long long insize_src, const byte* const __restrict__ input_resi, byte* const __restrict__ output, long long* const __restrict__ g_outsize)
{
  // allocate shared memory buffer
  __shared__ long long chunk [3 * (CS / sizeof(long long))];
  const int last = 3 * (CS / sizeof(long long)) - 2 - WS;

  // create the 3 shared memory buffers
  byte* in = (byte*)&chunk[0 * (CS / sizeof(long long))];
  byte* out = (byte*)&chunk[1 * (CS / sizeof(long long))];
  byte* temp = (byte*)&chunk[2 * (CS / sizeof(long long))];

  const int m = 23;  // mantissa bits

  const long long outsize = insize_src;
  const long long chunks = (outsize + CS - 1) / CS;  // round up

  // resi input header
  long long* const head_in_resi = (long long*)input_resi;
  int* const head_in_resi_eb = (int*)&head_in_resi[1];
  if (insize_src != head_in_resi[0]) {printf("ERROR: src orig size %lld doesn't match resi original size %lld \n\n", insize_src, head_in_resi[0]); __trap();}
  const int dest_eb_e = head_in_resi_eb[0];
  const int dest_thr_e = dest_eb_e + (m + 1);  // biased exponent of threshold
  const int dest_eb_offs = (dest_thr_e << m) - (1 << m);  // offset for lossless encoding

  // initialize resi input
  unsigned short* const size_in_resi = (unsigned short*)&head_in_resi[2];
  byte* const data_in_resi = (byte*)&size_in_resi[chunks];

  // loop over chunks
  const int tid = threadIdx.x;
  long long prevChunkID = 0;
  long long prevOffset_resi = 0;
  do {
    // assign work dynamically
    if (tid == 0) chunk[last] = atomicAdd(&g_chunk_counter, 1LL);
    __syncthreads();  // chunk[last] produced, chunk consumed

    // terminate if done
    const long long chunkID = chunk[last];
    const long long base = chunkID * CS;
    if (base >= outsize) break;

    // resi: compute sum of all prior csizes (start where left off in previous iteration)
    long long sum_resi = 0;
    for (long long i = prevChunkID + tid; i < chunkID; i += TPB) {
      sum_resi += (long long)size_in_resi[i];
    }
    int csize_resi = (int)size_in_resi[chunkID];
    const long long in_resi_offs = prevOffset_resi + block_sum_reduction(sum_resi, &chunk[last + 1]);
    prevOffset_resi = in_resi_offs;
    prevChunkID = chunkID;

    // load residual chunk
    g2s(in, &data_in_resi[in_resi_offs], csize_resi, out);
    byte* tmp = in; in = out; out = tmp;
    __syncthreads();  // chunk produced, chunk[last] consumed

    const int osize = (int)min((long long)CS, outsize - base);

    // decode
    if (csize_resi < osize) {
      byte* tmp;
      tmp = in; in = out; out = tmp;
      d_iRZE_1(csize_resi, in, out, temp);
      __syncthreads();
      tmp = in; in = out; out = tmp;
      d_iBIT_4(csize_resi, in, out, temp);
      __syncthreads();
      tmp = in; in = out; out = tmp;
      d_iDIFFMS_4(csize_resi, in, out, temp);
      __syncthreads();
    }
    if (csize_resi != osize) {printf("ERROR: csize_resi %d doesn't match osize %d in chunk %lld\n\n", csize_resi, osize, chunkID); __trap();}

    {
      // move out (residual) to temp
      byte* tmp;
      tmp = out; out = temp; temp = tmp;

      // load src chunk
      long long* const input_l = (long long*)&input_src[base];
      long long* const out_l = (long long*)out;
      for (int i = tid; i < osize / 8; i += TPB) {
        out_l[i] = input_l[i];
      }
      const int extra = osize % 8;
      if (tid < extra) out[(long long)osize - (long long)extra + (long long)tid] = input_src[base + (long long)osize - (long long)extra + (long long)tid];
      __syncthreads();

      // apply residual and dequantize
      tmp = in; in = out; out = tmp;
      d_iRESI_4(csize_resi, in, out, temp, dest_eb_e, dest_thr_e, dest_eb_offs);
      __syncthreads();
    }

    // write to output
    long long* const output_l = (long long*)&output[base];
    long long* const out_l = (long long*)out;
    for (int i = tid; i < osize / 8; i += TPB) {
      output_l[i] = out_l[i];
    }
    const int extra = osize % 8;
    if (tid < extra) output[base + osize - extra + tid] = out[osize - extra + tid];
    __syncthreads();  // prevent data race when chunk[last] is in the out buffer
  } while (true);

  if ((blockIdx.x == 0) && (tid == 0)) {
    *g_outsize = outsize;
  }
}


static inline void d_residual_apply(const byte* const d_src, const long long d_srcsize, const byte* const d_residual, byte* const d_dest, long long* const d_destsize, const int blocks)
{
  d_reset<<<1, 1>>>();
  d_residual_apply_kernel<<<blocks, TPB>>>(d_src, d_srcsize, d_residual, d_dest, d_destsize);
}

