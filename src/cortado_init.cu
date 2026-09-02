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

#include "include/d_client_encode.h"
#include "include/d_s2c.h"
#include "include/d_server_encode.h"
#include "include/preprocessors/d_client_QUANT_IABS_0_f32.h"
#include "include/preprocessors/include/INOA_f32.h"
#include "include/verifiers/MAXABS_f32.h"

int main(int argc, char *argv[]) {
  printf("Cortado v%s: Initial ", VERSION);
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
#ifdef RECV
  std::string init_filename;
  parser.add_argument("init_file")
      .help("the compressed initial file produced by an initial send")
      .store_into(init_filename);
#endif
#endif

#ifdef SEND
  parser.add_argument("request_eb")
      .help("the requested error bound; must be >= stored_eb")
      .nargs(1)
      .scan<'g', float>();

  std::string compressed_output_filename;
  parser.add_argument("-c", "--compressed_output")
      .help("if specified, outputs compressed initial file at requested error "
            "bound")
      .store_into(compressed_output_filename);
#endif

#ifdef RECV
  std::string decompressed_output_filename;
  parser.add_argument("-d", "--decompressed_output")
      .help("if specified, outputs decompressed file")
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
  float request_eb = parser.get<float>("request_eb");
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

#ifdef BENCHMARK
  if (request_eb < stored_eb) {
    fprintf(stderr,
            "ERROR: request eb must be greater than or equal to server eb\n\n");
    throw std::runtime_error("LC error");
  }
  printf("stored eb:  %lf\n", stored_eb);
#endif
#ifdef SEND
  printf("request eb: %lf\n", request_eb);
#endif
  if (runs > 1) {
    perf = true;
    printf("runs: %d\n", runs);
  }
  if (!perf) {
    runs = 1;
  }

  GPUTimer dtimer;

#ifdef BENCHMARK
  // read original input
  FILE *const fin = fopen(input_filename.c_str(), "rb");
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
    request_eb *= (maxf - minf);
    printf("adjusted stored eb:  %lf\n", stored_eb);
    printf("adjusted request eb: %lf\n", request_eb);
  }

  // setup: create server-stored data
  byte *d_encoded;
  cudaMalloc((void **)&d_encoded, maxsize);
  long long *d_encsize;
  cudaMalloc((void **)&d_encsize, sizeof(long long));
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
    request_eb *= noa_range;
  }

  // assuming ebs are sorted
  float server_eb_idx = -1;
  for (int i = 0; i < server_ebs.size(); i++) {
    if (request_eb >= server_ebs[i]) {
      server_eb_idx = i;
      break;
    }
  }
  if (server_eb_idx == -1) {
    fprintf(stderr,
            "ERROR: request eb (%lf) is too tight to be fulfilled by server "
            "(tightest eb: %lf)\n",
            request_eb, server_ebs.back());
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
  // read requant file
  FILE *const fin = fopen(init_filename.c_str(), "rb");
  fseek(fin, 0, SEEK_END);
  const long long fsize = ftell(fin);
  if (fsize <= 0) {
    fprintf(stderr, "ERROR: input file too small\n\n");
    throw std::runtime_error("LC error");
  }
  byte *const requant = new byte[fsize];
  fseek(fin, 0, SEEK_SET);
  const long long reqsize = fread(requant, 1, fsize, fin);
  assert(reqsize == fsize);
  fclose(fin);

  long long insize = ((long long *)requant)[0];

  printf("original size:        %lld bytes\n", insize);
  printf("compressed init size: %lld bytes\n", reqsize);

  const long long chunks = (insize + CS - 1) / CS; // round up
  const long long maxsize =
      2 * sizeof(long long) + chunks * sizeof(short) + chunks * CS;

  byte *d_requant;
  cudaMalloc((void **)&d_requant, reqsize);
  cudaMemcpy(d_requant, requant, reqsize, cudaMemcpyHostToDevice);
  CheckCuda(__LINE__);

  delete[] requant;
#endif

  /*
   * init send
   *   1) decompress and dequantize stored file
   *   2) requantize to requested eb
   *   3) compress to send
   */

#ifdef SEND
  printf("\n[init send]\n");

  byte *drequant;
  cudaMallocHost((void **)&drequant, maxsize);
  byte *d_requant;
  cudaMalloc((void **)&d_requant, maxsize);
  long long *d_reqsize;
  cudaMalloc((void **)&d_reqsize, sizeof(long long));
  CheckCuda(__LINE__);

  float send_init_runtimes[runs];

  if (perf) {
    // warm up
    byte *d_requant_dummy;
    cudaMalloc((void **)&d_requant_dummy, maxsize);
    long long *d_reqsize_dummy;
    cudaMalloc((void **)&d_reqsize_dummy, sizeof(long long));
    d_s2c_encode(d_encoded, d_requant_dummy, d_reqsize_dummy, request_eb,
                 chunks, blocks);
    cudaFree(d_requant_dummy);
    cudaFree(d_reqsize_dummy);
    CheckCuda(__LINE__);
  }

  for (int run = 0; run < runs; run++) {
    dtimer.start();
    d_s2c_encode(d_encoded, d_requant, d_reqsize, request_eb, chunks, blocks);
    send_init_runtimes[run] = dtimer.stop();
    CheckCuda(__LINE__);
  }

  float send_runtime = median(send_init_runtimes, runs);

  cudaFree(d_encoded);
  CheckCuda(__LINE__);

  // get encoded GPU result
  long long dreqsize = 0;
  cudaMemcpy(&dreqsize, d_reqsize, sizeof(long long), cudaMemcpyDeviceToHost);
  cudaMemcpy(drequant, d_requant, dreqsize, cudaMemcpyDeviceToHost);
  cudaFree(d_reqsize);
  CheckCuda(__LINE__);

  printf("encoded size:    %lld bytes\n", dreqsize);

  const float CR = (100.0 * dreqsize) / insize;
  printf("comp ratio:      %.2f%% %.3fx\n", CR, 100.0 / CR);

  if (perf) {
    printf("send runtime:    %.6f s\n", send_runtime);
    const float throughput = insize * 0.000000001 / send_runtime;
    printf("send throughput: %.3f Gbytes/s\n", throughput);
  }

  if (!compressed_output_filename.empty()) {
    write_to_file(dreqsize, drequant, compressed_output_filename);
    printf("wrote compressed initial file to %s\n",
           compressed_output_filename.c_str());
  }

  cudaFreeHost(drequant);
#endif

  /*
   * init recv
   *   - client-side decompression and dequantization
   */

#ifdef RECV
  printf("\n[init recv]\n");

  byte *ddecoded;
  cudaMallocHost((void **)&ddecoded, maxsize);
  byte *d_decoded;
  cudaMalloc((void **)&d_decoded, maxsize);
  long long *d_decsize;
  cudaMalloc((void **)&d_decsize, sizeof(long long));
  CheckCuda(__LINE__);

  float recv_runtimes[runs];

  if (perf) {
    // warm up
    byte *d_decoded_dummy;
    cudaMalloc((void **)&d_decoded_dummy, maxsize);
    long long *d_decsize_dummy;
    cudaMalloc((void **)&d_decsize_dummy, sizeof(long long));
    d_client_decode(d_requant, d_decoded_dummy, d_decsize_dummy, blocks);
    cudaFree(d_decoded_dummy);
    cudaFree(d_decsize_dummy);
    CheckCuda(__LINE__);
  }

  for (int run = 0; run < runs; run++) {
    dtimer.start();
    d_client_decode(d_requant, d_decoded, d_decsize, blocks);
    recv_runtimes[run] = dtimer.stop();
    CheckCuda(__LINE__);
  }

  float recv_runtime = median(recv_runtimes, runs);

  long long ddecsize = 0;
  cudaMemcpy(&ddecsize, d_decsize, sizeof(long long), cudaMemcpyDeviceToHost);
  cudaMemcpy(ddecoded, d_decoded, ddecsize, cudaMemcpyDeviceToHost);
  CheckCuda(__LINE__);

#ifdef BENCHMARK
  if (verify) {
    if (ddecsize != insize) {
      fprintf(stderr,
              "ERROR: decoded size of %lld does not match original insize of "
              "%lld\n",
              ddecsize, insize);
      throw std::runtime_error("LC error");
    }

    MAXABS_f32(insize, ddecoded, input, request_eb);

    byte *ddequant;
    cudaMallocHost((void **)&ddequant, maxsize);
    byte *d_dequant;
    cudaMalloc((void **)&d_dequant, maxsize);
    cudaMemcpy(d_dequant, input, insize, cudaMemcpyHostToDevice);
    CheckCuda(__LINE__);

    long long ddequantsize = insize;

    d_client_QUANT_IABS_0_f32(ddequantsize, d_dequant, request_eb);
    d_client_iQUANT_IABS_0_f32(ddequantsize, d_dequant);

    // get GPU result
    cudaMemcpy(ddequant, d_dequant, ddequantsize, cudaMemcpyDeviceToHost);
    cudaFree(d_dequant);
    CheckCuda(__LINE__);

    verify_match(ddequantsize, ddequant, ddecoded);

    cudaFreeHost(ddequant);
  }
#endif

  cudaFree(d_requant);
  cudaFree(d_decsize);
  cudaFree(d_decoded);
  CheckCuda(__LINE__);

  printf("decoded size:    %lld bytes\n", ddecsize);

  if (perf) {
    printf("recv runtime:    %.6f s\n", recv_runtime);
    const float throughput = insize * 0.000000001 / recv_runtime;
    printf("recv throughput: %.3f Gbytes/s\n", throughput);
  }

  if (!decompressed_output_filename.empty()) {
    write_to_file(ddecsize, ddecoded, decompressed_output_filename);
    printf("wrote decompressed file to %s\n",
           decompressed_output_filename.c_str());
  }

  cudaFreeHost(ddecoded);
#endif

#ifdef BENCHMARK
  delete[] input;
#endif
  return 0;
}
