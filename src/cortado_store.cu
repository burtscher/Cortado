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

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

#include <cuda.h>
#include <cuda/atomic>
#include <cuda/std/limits>

#include "include/argparse/argparse.hpp"

#include "include/consts.h"
#include "include/macros.h"
#include "include/util.h"

#include "include/d_server_encode.h"
#include "include/preprocessors/include/INOA_f32.h"
#include "include/verifiers/MAXABS_f32.h"

int main(int argc, char *argv[]) {
  printf("Cortado v%s: Server Encode and Store", VERSION);
#ifdef BENCHMARK
  printf(" (Benchmark Mode)");
#endif
  printf("\n");
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

  std::string input_filename;
  parser.add_argument("input")
      .help("the original input file")
      .store_into(input_filename);

  parser.add_argument("errorbounds")
      .help("list of error bounds to encode and store")
      .nargs(argparse::nargs_pattern::at_least_one)
      .scan<'g', float>();

  std::string output_dir;
  parser.add_argument("-o", "--output_dir")
      .help("if specified, outputs server storage to a directory")
      .store_into(output_dir);

  bool abs;
  parser.add_argument("-a", "--abs")
      .help("use absolute (ABS) error bound mode instead of "
            "normalized-absolute (NOA)")
      .flag()
      .store_into(abs);

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

  bool verify;
  parser.add_argument("-v", "--verify")
      .help("verify error bound of each stored file")
      .flag()
      .store_into(verify);

  try {
    parser.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << "ARGPARSE ERROR: " << err.what() << std::endl;
    std::cerr << parser;
    return 1;
  }

  printf("[info]\n");

  // get GPU info
  cudaSetDevice(0);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {
    fprintf(stderr, "ERROR: no CUDA capable device detected\n");
    throw std::runtime_error("LC error");
  }
  printf("GPU: %s\n", deviceProp.name);
  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  const int blocks = SMs * (mTpSM / TPB);
  CheckCuda(__LINE__);

  // clean up error bounds
  std::vector<float> ebs = parser.get<std::vector<float>>("errorbounds");
  std::sort(ebs.begin(), ebs.end(), std::greater<>());
  ebs.erase(std::unique(ebs.begin(), ebs.end()),
            ebs.end()); // unique items move to front, erase the tail

  printf("%s error bounds (%ld):\n", abs ? "ABS" : "NOA", ebs.size());
  for (const float &eb : ebs) {
    printf("  %lf\n", eb);
  }
  if (runs > 1) {
    perf = true;
    printf("runs: %d\n", runs);
  }
  if (!perf) {
    runs = 1;
  }

  if (!output_dir.empty()) {
    try {
      std::filesystem::remove_all(
          output_dir); // remove old storage if it exists
      std::filesystem::create_directories(output_dir);
    } catch (const std::filesystem::filesystem_error &e) {
      std::cerr << "FILESYSTEM ERROR: " << e.what() << std::endl;
      return 1;
    }
  }

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

  float minf, maxf;
  d_INOA_find_range(insize, d_input, minf, maxf);
  printf("NOA min: %lf\nNOA max: %lf\n", minf, maxf);

  // allocate GPU memory
  byte *dencoded;
  cudaMallocHost((void **)&dencoded, maxsize);
  byte *d_encoded;
  cudaMalloc((void **)&d_encoded, maxsize);
  long long *d_encsize;
  cudaMalloc((void **)&d_encsize, sizeof(long long));
  CheckCuda(__LINE__);

  GPUTimer dtimer;

  for (int i = 0; i < ebs.size(); i++) {
    const float orig_eb = ebs[i]; // hold user-provided eb
    if (!abs) {
      ebs[i] = (maxf - minf) * orig_eb; // adjust eb in-place
    }
    const float eb = ebs[i];
    const float client_eb = eb;
    const float server_eb = eb / 2;
    printf("\n[store eb %d/%ld] orig: %lf, client: %lf, server: %lf\n", i + 1,
           ebs.size(), orig_eb, client_eb, server_eb);

    float runtimes[runs];

    if (perf) {
      d_server_encode(d_input, insize, d_encoded, d_encsize, eb, chunks,
                      blocks);
      CheckCuda(__LINE__);
    }

    for (int run = 0; run < runs; run++) {
      dtimer.start();
      d_server_encode(d_input, insize, d_encoded, d_encsize, eb, chunks,
                      blocks);
      runtimes[run] = dtimer.stop();
      CheckCuda(__LINE__);
    }

    float runtime = median(runtimes, runs);

    if (verify) {
      byte *ddecoded;
      cudaMallocHost((void **)&ddecoded, maxsize);
      byte *d_decoded;
      cudaMalloc((void **)&d_decoded, maxsize);
      long long *d_decsize;
      cudaMalloc((void **)&d_decsize, sizeof(long long));
      CheckCuda(__LINE__);

      d_server_decode(d_encoded, d_decoded, d_decsize, blocks);

      // get GPU result
      long long ddecsize = 0;
      cudaMemcpy(&ddecsize, d_decsize, sizeof(long long),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(ddecoded, d_decoded, ddecsize, cudaMemcpyDeviceToHost);
      cudaFree(d_decoded);
      cudaFree(d_decsize);
      CheckCuda(__LINE__);

      if (ddecsize != insize) {
        fprintf(stderr,
                "ERROR: decoded size of %lld does not match original insize of "
                "%lld\n",
                ddecsize, insize);
        throw std::runtime_error("LC error");
      }

      MAXABS_f32(insize, ddecoded, input, client_eb);

      cudaFreeHost(ddecoded);
    }

    // get encoded GPU result
    long long dencsize = 0;
    cudaMemcpy(&dencsize, d_encsize, sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(dencoded, d_encoded, dencsize, cudaMemcpyDeviceToHost);
    CheckCuda(__LINE__);

    printf("encoded size: %lld bytes\n", dencsize);

    const float CR = (100.0 * dencsize) / insize;
    printf("comp ratio:   %.2f%%  %.3fx\n", CR, 100.0 / CR);

    if (perf) {
      printf("runtime:      %.6f s\n", runtime);
      float throughput = insize * 0.000000001 / runtime;
      printf("throughput:   %.3f Gbytes/s\n", throughput);
    }

    if (!output_dir.empty()) {
      std::string fout_name =
          (std::filesystem::path(output_dir) /
           (std::filesystem::path(input_filename).filename().string() +
            ".stor." + pad_zeros(i, 2)))
              .string();
      write_to_file(dencsize, dencoded, fout_name);
      printf("wrote compressed server file to %s\n", fout_name.c_str());
    }
  }

  if (!output_dir.empty()) {
    std::string meta_filename =
        write_server_metadata(output_dir, input_filename, ebs, (maxf - minf));
    printf("wrote server metadata file to %s\n", meta_filename.c_str());
  }

  cudaFree(d_input);
  cudaFree(d_encoded);
  cudaFree(d_encsize);
  CheckCuda(__LINE__);

  delete[] input;
  cudaFreeHost(dencoded);
  return 0;
}
