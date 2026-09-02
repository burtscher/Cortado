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


#ifndef UTIL_H
#define UTIL_H

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <ctype.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#define VERSION "1.0"

template <typename T>
void write_to_file(const long long size, const T *const data,
                   std::string filename) {
  FILE *fout = fopen(filename.c_str(), "wb");
  assert(fout != NULL);
  long long fsize = fwrite(data, sizeof(T), size, fout);
  assert(size == fsize);
  fclose(fout);
}

std::string write_server_metadata(const std::string output_dir,
                           const std::string input_filename,
                           const std::vector<float> ebs, const float range) {
  assert(std::filesystem::exists(output_dir));
  std::string output_filename =
      (std::filesystem::path(output_dir) / ".stor.meta").string();

  const int len = (2 * sizeof(int)) + (input_filename.size() * sizeof(char)) +
                  (ebs.size() * sizeof(float)) + sizeof(float);

  byte *const out = new byte[len];

  int *out_i = (int *)out;
  out_i[0] = input_filename.size();
  out_i[1] = ebs.size();

  float *out_f = (float *)&out_i[2];
  for (int i = 0; i < ebs.size(); i++) {
    out_f[i] = ebs[i];
  }
  out_f[ebs.size()] = range;

  char *out_s = (char *)&out_f[ebs.size() + 1];
  memcpy(out_s, input_filename.c_str(), input_filename.size() * sizeof(char));

  write_to_file(len, out, output_filename);
  delete [] out;

  return output_filename;
}

void read_server_metadata(const std::string server_dir,
                           std::string &input_filename,
                           std::vector<float> &ebs, float &range) {
  std::string meta_filename =
      (std::filesystem::path(server_dir) / ".stor.meta").string();

  FILE *fin = fopen(meta_filename.c_str(), "rb");
  fseek(fin, 0, SEEK_END);
  const long long fsize = ftell(fin);
  if (fsize <= 0) {
    fprintf(stderr, "ERROR: input file too small\n\n");
    throw std::runtime_error("server metadata read error");
  }
  byte *const in = new byte[fsize];
  fseek(fin, 0, SEEK_SET);
  long long insize = fread(in, sizeof(byte), fsize, fin);
  assert(insize == fsize);
  fclose(fin);

  int *in_i = (int *)in;
  const int name_len = in_i[0];
  const int ebs_count = in_i[1];

  float *in_f = (float *)&in_i[2];
  ebs.resize(ebs_count);
  for (int i = 0; i < ebs_count; i++) {
    ebs[i] = in_f[i];
  }
  range = in_f[ebs_count];

  char *in_s = (char *)&in_f[ebs_count + 1];
  input_filename.assign(in_s, name_len);

  delete[] in;
}

std::string pad_zeros(const int value, const int width) {
  std::ostringstream ss;
  ss << std::setfill('0') << std::setw(width) << value;
  return ss.str();
}

template <typename T>
void verify_match(const long long size, T *expected, T *actual) {
  for (long long i = 0; i < size; i++) {
    if (expected[i] != actual[i]) {
      printf("ERROR: bit-for-bit verification failed: Mismatch at %lld/%lld. "
             "Expected %d but got "
             "%d\n",
             i, size, expected[i], actual[i]);
      exit(-1);
    }
  }
  printf("bit-for-bit verification passed\n");
}

static float median(float array[], const int n) {
  float median = 0;
  std::sort(array, array + n);
  if (n % 2 == 0)
    median = (array[(n - 1) / 2] + array[n / 2]) / 2.0;
  else
    median = array[n / 2];
  return median;
}

#ifdef USE_GPU

struct GPUTimer {
  cudaEvent_t beg, end;
  GPUTimer() {
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
  }
  ~GPUTimer() {
    cudaEventDestroy(beg);
    cudaEventDestroy(end);
  }
  void start() { cudaEventRecord(beg, 0); }
  double stop() {
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float ms;
    cudaEventElapsedTime(&ms, beg, end);
    return 0.001 * ms;
  }
};

static void CheckCuda(const int line) {
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d on line %d: %s\n\n", e, line,
            cudaGetErrorString(e));
    throw std::runtime_error("LC error");
  }
}

#endif /* USE_GPU */

#endif /* UTIL_H */
