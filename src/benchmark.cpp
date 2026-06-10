#include <benchmark/benchmark.h>
#include <cstdlib>
#include <filesystem>
#include <math.h>
#define N 1000
#define P 500

extern "C" {
#include "c_shrinkr.h"
}

namespace fs = std::filesystem;

void fill_random_normal(double* data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        // Box-Muller transform to generate standard normal random variables
        double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
        double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
        data[i] = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    }
}

int load_binary_doubles_c(const char* filename, double* dest, size_t num_elements) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
      printf("DEBUG: Failed to open file '%s'. Check if the file exists and the path is correct.\n", filename);
      return 1;
    };
    size_t elements_read = fread(dest, sizeof(double), num_elements, file);
    fclose(file);
    if (elements_read != num_elements) {
      printf("DEBUG: Expected %zu elements, but read %zu elements.\n", num_elements, elements_read);
      return 2;
    }
    return 0;
}

inline int read_benchmark_file(const std::string& prefix, size_t n, size_t p, size_t num_elements, double* dest) {
    fs::path data_dir = fs::path("..") / ".devel" / "bench" / "data";
    std::string file_name = prefix + "_" + std::to_string(p) + "_" + std::to_string(n) + ".bin";
    fs::path full_path = data_dir / file_name;
    return load_binary_doubles_c(full_path.string().c_str(), dest, num_elements);
}


static void BM_OAS(benchmark::State &state) {
  double *sample_cov = (double *) malloc(P * P * sizeof(double));
  double *sample_cov_star = (double *) malloc(P * P * sizeof(double));

  int rescode = read_benchmark_file("sample_cov", N, P, P * P, sample_cov);
  if (rescode) {
    state.SkipWithError("Error reading sample_cov file. Check file size.");
    free(sample_cov);
    free(sample_cov_star);
    return;
  }

  for (auto _ : state) {
    C_OAS(sample_cov, sample_cov_star, N, P);
    benchmark::ClobberMemory();
  }

  free(sample_cov);
  free(sample_cov_star);
}


static void BM_LWAnalytical(benchmark::State &state) {
  double *lam = (double *) malloc(P * sizeof(double));
  double *lam_star = (double *) malloc(P * sizeof(double));

  int rescode = read_benchmark_file("eigenvalues", N, P, P, lam);
  if (rescode) {
    state.SkipWithError("Error reading eigenvalues file.");
    free(lam);
    free(lam_star);
    return;
  }

  for (auto _ : state) {
    C_LWAnalytical(lam, lam_star, N, P, 1e-8);
    benchmark::ClobberMemory();
  }

  free(lam);
  free(lam_star);
}

static void BM_LWLinear(benchmark::State &state) {
  double *data = (double *) malloc(N * P * sizeof(double));
  double *sample_cov_star = (double *) malloc(P * P * sizeof(double));

  int rescode = read_benchmark_file("data", N, P, N * P, data);
  if (rescode) {
    state.SkipWithError("Error reading data file. Check file size.");
    free(data);
    free(sample_cov_star);
    return;
  }

  for (auto _ : state) {
    C_LWLinear(data, sample_cov_star, N, P);
    benchmark::ClobberMemory();
  }

  free(data);
  free(sample_cov_star);
}

static void BM_DEAL(benchmark::State &state) {
  double *base_evals = (double *) malloc(P * sizeof(double));

  int rescode = read_benchmark_file("eigenvalues", N, P, P, base_evals);
  if (rescode) {
    state.SkipWithError("Error reading eigenvalues file. Check file sizes.");
    free(base_evals);
    return;
  }

  double *surr_evals = (double *) malloc(P * sizeof(double));
  double *z_vec = (double *) malloc(P * sizeof(double));

  // Compute surrogate eigenvalues
  C_LWAnalytical(base_evals, surr_evals, N, P, 1e-8);

  // Genearate random direction
  fill_random_normal(z_vec, P);


  double gamma_min = 1e-4;
  double gamma_max = 1e4;
  double start_value = 1;

  for (auto _ : state) {
    C_DEAL(base_evals, surr_evals, z_vec, gamma_min, gamma_max, N, P);
    benchmark::ClobberMemory();
  }

  free(base_evals);
  free(surr_evals);
  free(z_vec);
}


BENCHMARK(BM_OAS);
BENCHMARK(BM_LWAnalytical);
BENCHMARK(BM_LWLinear);
BENCHMARK(BM_DEAL);

BENCHMARK_MAIN();

