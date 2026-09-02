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

#if !defined(BENCHMARK) && !defined(SEND) && !defined(RECV)
#define BENCHMARK // fallback to benchmark mode
#endif

#ifdef BENCHMARK
#define SEND
#define RECV
#endif

#include <cassert>
#include <cstdio>
#include <stdexcept>

#include <cuda.h>
#include <cuda/atomic>
#include <cuda/std/limits>

#include "include/argparse/argparse.hpp"

#include "include/consts.h"
#include "include/macros.h"
#include "include/util.h"

#include "include/preprocessors/include/INOA_f32.h"

#include "include/d_client_encode.h"
#include "include/d_residual_encode.h"
#include "include/d_s2c.h"
#include "include/d_server_encode.h"
#include "include/preprocessors/d_client_QUANT_IABS_0_f32.h"
#include "include/preprocessors/d_residual_QUANT_IABS_0_f32.h"
#include "include/preprocessors/d_server_QUANT_IABS_0_f32.h"
#include "include/verifiers/MAXABS_f32.h"

int main(int argc, char *argv[]) {
  printf("Cortado v%s: Residual ", VERSION);
#ifdef BENCHMARK
  printf("Send/Receive (Benchmark Mode)\n");
#else
#ifdef SEND
  printf("Send\n");
#endif
#ifdef RECV
  printf("Receive\n");
#endif
#endif
  printf("Copyright 2026 Texas State University\n\n");

  argparse::ArgumentParser parser(argv[0], VERSION,
                                  argparse::default_arguments::none);

  // manually re-add the help argument
  parser.add_argument("-h", "--help")
      .action([&](const std::string &) {
        std::cout << parser.help().str();
        std::exit(0);
      })
      .default_value(false)
      .help("shows help message and exits")
      .implicit_value(true);

#ifdef BENCHMARK
  std::string input_filename;
  parser.add_argument("input")
      .help("the original input file")
      .store_into(input_filename);

  parser.add_argument("stored_eb")
      .help("the stored error bound from which to fulfill the request")
      .nargs(1)
      .scan<'g', float>();
#else
#ifdef SEND
  std::string server_dir;
  parser.add_argument("server_dir")
      .help("path to directory containing stored server files")
      .store_into(server_dir);
#endif
#endif

#ifdef SEND
  parser.add_argument("src_eb")
      .help("the client's current error bound; must be >= dst_eb")
      .nargs(1)
      .scan<'g', float>();

  parser.add_argument("dst_eb")
      .help("the requested error bound to refine to"
#ifdef BENCHMARK
            " ; must be >= stored_eb"
#endif
            )
      .nargs(1)
      .scan<'g', float>();
#endif

#if !defined(BENCHMARK) && defined(RECV)
  std::string src_filename;
  parser.add_argument("src_file")
      .help("the client's current file (decompressed by initial receive)")
      .store_into(src_filename);

  std::string resi_filename;
  parser.add_argument("resi_file")
      .help("the compressed residual file produced by a residual send")
      .store_into(resi_filename);
#endif

#ifdef SEND
  std::string compressed_output_filename;
  parser.add_argument("-c", "--compressed_output")
      .help("if specified, outputs compressed residual file")
      .store_into(compressed_output_filename);
#endif

#ifdef RECV
  std::string decompressed_output_filename;
  parser.add_argument("-d", "--decompressed_output")
      .help("if specified, outputs the final decompressed file with the "
            "residual applied")
      .store_into(decompressed_output_filename);
#endif

#if defined(SEND)
  bool abs;
  parser.add_argument("-a", "--abs")
      .help("use absolute (ABS) error bound mode instead of "
            "normalized-absolute (NOA)")
      .flag()
      .store_into(abs);
#endif

  bool perf;
#ifdef BENCHMARK
  perf = true;
#else
  parser.add_argument("-p", "--perf")
      .help("run warmups and record performance metrics")
      .flag()
      .store_into(perf);
#endif

  int runs;
  parser.add_argument("-r", "--runs")
      .help("number of timed runs to perform; output reports the "
            "median of all times")
      .scan<'i', int>()
      .default_value(1)
      .store_into(runs);

#ifdef BENCHMARK
  // only benchmark mode has access to the original input to verify against
  bool verify;
  parser.add_argument("-v", "--verify")
      .help("verify error bound and bit-for-bit parity of recieved file")
      .flag()
      .store_into(verify);
#endif

  try {
    parser.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << "ARGPARSE ERROR: " << err.what() << std::endl;
    std::cerr << parser;
    return 1;
  }

  printf("[info]\n");

#ifdef BENCHMARK
  float stored_eb = parser.get<float>("stored_eb");
#endif
#ifdef SEND
  float src_eb = parser.get<float>("src_eb");
  float dst_eb = parser.get<float>("dst_eb");
#endif

  // get GPU info
  cudaSetDevice(0);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {
    fprintf(stderr, "ERROR: no CUDA capable device detected\n\n");
    throw std::runtime_error("LC error");
  }
  printf("GPU: %s\n", deviceProp.name);
  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  const int blocks = SMs * (mTpSM / TPB);
  CheckCuda(__LINE__);

#ifdef SEND
  printf("%s error bound mode\n", abs ? "ABS" : "NOA");
#endif

#ifdef SEND
  if (src_eb < dst_eb) {
    fprintf(stderr,
            "ERROR: src eb must be greater than or equal to dest eb\n\n");
    throw std::runtime_error("LC error");
  }
#ifdef BENCHMARK
  if (dst_eb < stored_eb) {
    fprintf(stderr, "ERROR: dest and src eb must be greater than or equal to "
                    "stored eb\n\n");
    throw std::runtime_error("LC error");
  }
  printf("stored eb: %lf\n", stored_eb);
#endif
  printf("src eb:    %lf\n", src_eb);
  printf("dest eb:   %lf\n", dst_eb);
#endif
  if (runs > 1) {
    perf = true;
    printf("runs: %d\n", runs);
  }
  if (!perf) {
    runs = 1;
  }

  GPUTimer dtimer;

  /*
   * Read input file(s)
   */

#ifdef BENCHMARK
  // read input from file
  FILE *const fin = fopen(argv[1], "rb");
  fseek(fin, 0, SEEK_END);
  const long long fsize = ftell(fin);
  if (fsize <= 0) {
    fprintf(stderr, "ERROR: input file too small\n\n");
    throw std::runtime_error("LC error");
  }
  byte *const input = new byte[fsize];
  fseek(fin, 0, SEEK_SET);
  const long long insize = fread(input, 1, fsize, fin);
  assert(insize == fsize);
  fclose(fin);
  printf("original size: %lld bytes\n", insize);

  const long long chunks = (insize + CS - 1) / CS; // round up
  const long long maxsize =
      2 * sizeof(long long) + chunks * sizeof(short) + chunks * CS;

  byte *d_input;
  cudaMalloc((void **)&d_input, insize);
  cudaMemcpy(d_input, input, insize, cudaMemcpyHostToDevice);
  CheckCuda(__LINE__);

  if (!abs) {
    float minf, maxf;
    d_INOA_find_range(insize, d_input, minf, maxf);
    printf("NOA min: %lf\nNOA max: %lf\n", minf, maxf);
    stored_eb *= (maxf - minf);
    src_eb *= (maxf - minf);
    dst_eb *= (maxf - minf);
    printf("adjusted stored eb: %lf\n", stored_eb);
    printf("adjusted src eb:    %lf\n", src_eb);
    printf("adjusted dest eb:   %lf\n", dst_eb);
  }

  // setup: create server-stored data
  byte *d_encoded;
  cudaMalloc((void **)&d_encoded, maxsize);
  long long *d_encsize;
  cudaMalloc((void **)&d_encsize, sizeof(long long));
  CheckCuda(__LINE__);
  d_server_encode(d_input, insize, d_encoded, d_encsize, stored_eb, chunks,
                  blocks);
  cudaFree(d_input);
  cudaFree(d_encsize);
  CheckCuda(__LINE__);
#endif

#if !defined(BENCHMARK) && defined(SEND)
  std::string input_filename;
  std::vector<float> server_ebs;
  float noa_range;
  read_server_metadata(server_dir, input_filename, server_ebs, noa_range);

  printf("server error bounds (%ld):\n", server_ebs.size());
  for (const float &eb : server_ebs) {
    printf("  %lf\n", abs ? eb : (eb / noa_range));
  }

  if (!abs) {
    src_eb *= noa_range;
    dst_eb *= noa_range;
  }

  // assuming ebs are sorted
  float server_eb_idx = -1;
  for (int i = 0; i < server_ebs.size(); i++) {
    if (dst_eb >= server_ebs[i]) {
      server_eb_idx = i;
      break;
    }
  }
  if (server_eb_idx == -1) {
    fprintf(stderr,
            "ERROR: dst eb (%lf) is too tight to be fulfilled by server "
            "(tightest eb: %lf)\n",
            dst_eb, server_ebs.back());
    return -1;
  }

  printf("requantizing from server eb: %lf\n",
         abs ? server_ebs[server_eb_idx]
             : (server_ebs[server_eb_idx] / noa_range));

  std::string server_filename =
      (std::filesystem::path(server_dir) /
       (std::filesystem::path(input_filename).filename().string() + ".stor." +
        pad_zeros(server_eb_idx, 2)))
          .string();

  // read dencoded file
  FILE *const fin = fopen(server_filename.c_str(), "rb");
  fseek(fin, 0, SEEK_END);
  const long long fsize = ftell(fin);
  if (fsize <= 0) {
    fprintf(stderr, "ERROR: input file too small\n\n");
    throw std::runtime_error("LC error");
  }
  byte *const encoded = new byte[fsize];
  fseek(fin, 0, SEEK_SET);
  const long long encsize = fread(encoded, 1, fsize, fin);
  assert(encsize == fsize);
  fclose(fin);

  // read header to get insize of original file
  const long long insize = ((long long *)encoded)[0];

  printf("original size:    %lld bytes\n", insize);
  printf("server file size: %lld bytes\n", encsize);

  const long long chunks = (insize + CS - 1) / CS; // round up
  const long long maxsize =
      2 * sizeof(long long) + chunks * sizeof(short) + chunks * CS;

  // transfer to GPU
  byte *d_encoded;
  cudaMalloc((void **)&d_encoded, encsize);
  cudaMemcpy(d_encoded, encoded, encsize, cudaMemcpyHostToDevice);
  CheckCuda(__LINE__);

  delete[] encoded;
#endif

#if !defined(BENCHMARK) && defined(RECV)
  // read decompressed src file
  FILE *const fin_src = fopen(src_filename.c_str(), "rb");
  fseek(fin_src, 0, SEEK_END);
  const long long fsize_src = ftell(fin_src);
  if (fsize_src <= 0) {
    fprintf(stderr, "ERROR: input file too small\n\n");
    throw std::runtime_error("LC error");
  }
  byte *const src = new byte[fsize_src];
  fseek(fin_src, 0, SEEK_SET);
  const long long dsrcsize = fread(src, 1, fsize_src, fin_src);
  assert(dsrcsize == fsize_src);
  fclose(fin_src);

  const long long insize = dsrcsize;

  printf("decompressed src size:  %lld bytes\n", insize);

  const long long chunks = (insize + CS - 1) / CS; // round up
  const long long maxsize =
      2 * sizeof(long long) + chunks * sizeof(short) + chunks * CS;

  byte *d_src;
  cudaMalloc((void **)&d_src, dsrcsize);
  cudaMemcpy(d_src, src, dsrcsize, cudaMemcpyHostToDevice);
  CheckCuda(__LINE__);

  delete[] src;

  // read compressed residual file
  FILE *const fin_resi = fopen(resi_filename.c_str(), "rb");
  fseek(fin_resi, 0, SEEK_END);
  const long long fsize_resi = ftell(fin_resi);
  if (fsize_resi <= 0) {
    fprintf(stderr, "ERROR: input file too small\n\n");
    throw std::runtime_error("LC error");
  }
  byte *const residual = new byte[fsize_resi];
  fseek(fin_resi, 0, SEEK_SET);
  const long long residualsize = fread(residual, 1, fsize_resi, fin_resi);
  assert(residualsize == fsize_resi);
  fclose(fin_resi);

  printf("compressed resi size:   %lld bytes\n", residualsize);

  byte *d_residual;
  cudaMalloc((void **)&d_residual, residualsize);
  cudaMemcpy(d_residual, residual, residualsize, cudaMemcpyHostToDevice);
  CheckCuda(__LINE__);

  delete[] residual;
#endif

  /*
   * resi send
   *   1) decompress and dequantize stored file
   *   2) compute residual (src -> dst)
   *   3) compress to send
   */

#ifdef SEND
  printf("\n[resi send]\n");

  byte *dresidual;
  cudaMallocHost((void **)&dresidual, maxsize);
  byte *d_residual;
  cudaMalloc((void **)&d_residual, maxsize);
  long long *d_residualsize;
  cudaMalloc((void **)&d_residualsize, sizeof(long long));
  CheckCuda(__LINE__);

  float send_runtimes[runs];

  if (perf) {
    // warm up
    byte *d_residual_dummy;
    cudaMalloc((void **)&d_residual_dummy, maxsize);
    long long *d_residualsize_dummy;
    cudaMalloc((void **)&d_residualsize_dummy, sizeof(long long));
    d_residual_encode(d_encoded, d_residual_dummy, d_residualsize_dummy, src_eb,
                      dst_eb, chunks, blocks);
    cudaFree(d_residual_dummy);
    cudaFree(d_residualsize_dummy);
    CheckCuda(__LINE__);
  }

  for (int run = 0; run < runs; run++) {
    /*
     * resi recv
     *   - decompress and apply residual file to client's uncompressed src file
     */

    dtimer.start();
    d_residual_encode(d_encoded, d_residual, d_residualsize, src_eb, dst_eb,
                      chunks, blocks);
    send_runtimes[run] = dtimer.stop();
    CheckCuda(__LINE__);
  }

  // get encoded GPU result
  long long dresidualsize = 0;
  cudaMemcpy(&dresidualsize, d_residualsize, sizeof(long long),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(dresidual, d_residual, dresidualsize, cudaMemcpyDeviceToHost);
  cudaFree(d_residualsize);
  CheckCuda(__LINE__);

  printf("encoded size:    %lld bytes\n", dresidualsize);

  const float CR = (100.0 * dresidualsize) / insize;
  printf("comp ratio:      %.2f%% %7.3fx\n", CR, 100.0 / CR);

  const float send_runtime = median(send_runtimes, runs);
  if (perf) {
    printf("send runtime:    %.6f s\n", send_runtime);
    const float throughput = insize * 0.000000001 / send_runtime;
    printf("send throughput: %.3f Gbytes/s\n", throughput);
  }

  if (!compressed_output_filename.empty()) {
    write_to_file(dresidualsize, dresidual, compressed_output_filename);
    printf("wrote compressed residual file to %s\n",
           compressed_output_filename.c_str());
  }

  cudaFreeHost(dresidual);
#endif

  /*
   * resi recv
   *   - decompress and apply residual file to client's uncompressed src file
   */

#ifdef BENCHMARK
  // setup: prepare src file
  byte *d_src_enc;
  cudaMalloc((void **)&d_src_enc, maxsize);
  long long *d_src_encsize;
  cudaMalloc((void **)&d_src_encsize, sizeof(long long));
  CheckCuda(__LINE__);

  d_s2c_encode(d_encoded, d_src_enc, d_src_encsize, src_eb, chunks, blocks);
  cudaFree(d_encoded);
  cudaFree(d_src_encsize);

  // setup: uncompressed and dequantized src
  byte *d_src;
  cudaMalloc((void **)&d_src, maxsize);
  long long *d_srcsize;
  cudaMalloc((void **)&d_srcsize, sizeof(long long));
  CheckCuda(__LINE__);

  d_client_decode(d_src_enc, d_src, d_srcsize, blocks);

  long long dsrcsize;
  cudaMemcpy(&dsrcsize, d_srcsize, sizeof(long long), cudaMemcpyDeviceToHost);
  cudaFree(d_srcsize);
  cudaFree(d_src_enc);
  CheckCuda(__LINE__);
#endif

#ifdef RECV
  printf("\n[resi recv]\n");

  byte *d_dest;
  cudaMalloc((void **)&d_dest, maxsize);
  long long *d_destsize;
  cudaMalloc((void **)&d_destsize, sizeof(long long));
  CheckCuda(__LINE__);

  float recv_runtimes[runs];

  if (perf) {
    // warmup
    byte *d_dest_dummy;
    cudaMalloc((void **)&d_dest_dummy, maxsize);
    long long *d_destsize_dummy;
    cudaMalloc((void **)&d_destsize_dummy, sizeof(long long));
    d_residual_apply(d_src, dsrcsize, d_residual, d_dest_dummy,
                     d_destsize_dummy, blocks);
    cudaFree(d_dest_dummy);
    cudaFree(d_destsize_dummy);
    CheckCuda(__LINE__);
  }

  for (int run = 0; run < runs; run++) {
    dtimer.start();
    d_residual_apply(d_src, dsrcsize, d_residual, d_dest, d_destsize, blocks);
    recv_runtimes[run] = dtimer.stop();
    CheckCuda(__LINE__);
  }

  const float recv_runtime = median(recv_runtimes, runs);

  long long ddestsize;
  cudaMemcpy(&ddestsize, d_destsize, sizeof(long long), cudaMemcpyDeviceToHost);
  cudaFree(d_destsize);
  cudaFree(d_src);

  byte *ddest;
  cudaMallocHost((void **)&ddest, maxsize);

  // get GPU result
  cudaMemcpy(ddest, d_dest, ddestsize, cudaMemcpyDeviceToHost);
  CheckCuda(__LINE__);

#ifdef BENCHMARK
  if (verify) {
    if (ddestsize != insize) {
      fprintf(stderr,
              "ERROR: decoded size of %lld does not match original insize of "
              "%lld\n",
              ddestsize, insize);
      throw std::runtime_error("LC error");
    }

    byte *ddequant;
    cudaMallocHost((void **)&ddequant, maxsize);
    byte *d_dequant;
    cudaMalloc((void **)&d_dequant, maxsize);
    cudaMemcpy(d_dequant, input, insize, cudaMemcpyHostToDevice);
    CheckCuda(__LINE__);

    long long ddequantsize = insize;
    d_client_QUANT_IABS_0_f32(ddequantsize, d_dequant, dst_eb);
    d_client_iQUANT_IABS_0_f32(ddequantsize, d_dequant);

    // get GPU result
    cudaMemcpy(ddequant, d_dequant, ddequantsize, cudaMemcpyDeviceToHost);
    cudaFree(d_dequant);
    CheckCuda(__LINE__);

    MAXABS_f32(insize, ddest, input, dst_eb);
    verify_match(ddequantsize, ddequant, ddest);

    cudaFreeHost(ddequant);
  }
#endif

  cudaFree(d_dest);
  cudaFree(d_residual);
  CheckCuda(__LINE__);

  printf("decoded size:    %lld bytes\n", ddestsize);

  if (perf) {
    printf("recv runtime:    %.6f s\n", recv_runtime);
    const float throughput = insize * 0.000000001 / recv_runtime;
    printf("recv throughput: %.3f Gbytes/s\n", throughput);
  }

  if (!decompressed_output_filename.empty()) {
    write_to_file(ddestsize, ddest, decompressed_output_filename);
    printf("wrote decompressed file to %s\n",
           decompressed_output_filename.c_str());
  }

  cudaFreeHost(ddest);
#endif

#ifdef BENCHMARK
  delete[] input;
#endif
  return 0;
}
