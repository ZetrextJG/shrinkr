#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "c_shrinkr.h"
#define SQRT5 2.23606797749979
#define PARALLEL_THRESHOLD 200
#define DOUBLE_EPS 1e-12
#define SQUARE(x) (x * x)
#define MAX(x, y) ((x > y) ? x : y)
#define MIN(x, y) ((x > y) ? y : x)

#ifdef _OPENMP
#include <omp.h>
#endif

// Utilities functions

double relu(double x) {
  return x > 0.0 ? x : 0.0;
}

double clip(double x, double min, double max) {
  if (x < min) {
    return min;
  } else if (x > max) {
    return max;
  }
  return x;
}

double trace(const double * const matrix, size_t p) {
  double acc = 0;
  for (size_t i = 0; i < p; ++i) {
    acc += matrix[i*p + i];
  }
  return acc;
}

double traceS2divp2(const double * const matrix, size_t p) {
  // Computes tr(S @ S.T) / (p^2) for a symmetric matrix S
  double acc = 0;
  size_t max_iter = SQUARE(p);
  #pragma omp parallel for reduction(+:acc) if(max_iter >= PARALLEL_THRESHOLD)
  for (size_t i = 0; i < max_iter ; ++i) {
    acc += SQUARE(matrix[i]) / max_iter;
  }
  return acc;
}

void scalar_multiply(double * const data, size_t n, double scale) {
  #pragma omp parallel for if(n >= PARALLEL_THRESHOLD)
  for (size_t i = 0; i < n ; ++i) {
    data[i] = scale * data[i];
  }
}

// Main implementation
void C_OAS(
    const double * const sample_cov, // Sample covariance
    double * const sample_cov_star, // Shrunk covariance
    size_t n, // Number of samples used
    size_t p  // Dimensions of the sample_cov
) {
  size_t p2 = SQUARE(p);

  // Init the sample_cov_star matrix as sample_cov
  memcpy(sample_cov_star, sample_cov, p2 * sizeof(double));

  double alpha = traceS2divp2(sample_cov, p);
  double mu = trace(sample_cov, p) / p;
  double mu_squared = SQUARE(mu);

  double num = alpha + mu_squared;
  double denom = (n + 1) * (alpha - mu_squared / p);
  double shrinkage = denom < DOUBLE_EPS ? 1.0 : clip(num / denom, 0, 1);

  // Shrink the cov
  scalar_multiply(sample_cov_star, SQUARE(p), (1.0 - shrinkage));

  // Add on the diagonal
  double add_value = shrinkage * mu;
  for (size_t i = 0; i < p; ++i) {
    sample_cov_star[i*p + i] += add_value;
  }

  return ;
}


void C_LWAnalytical(
    const double * const lam, // Input data
    double * const lam_star, // Output data
    size_t n, // Number of sample used
    size_t p, // Dimensions of both x and y
    double eps // Epsilon value
) {
  double h = pow(n, - 1.0 / 3.0);
  double d_p = (double) p;
  double d_n = (double) n;
  double ratio = d_p / d_n;
  double c1 = 3.0 / (4.0 * SQRT5); // \frac{3}{4\sqrt{5}}

  size_t shift = n < p ? p - n : 0;
  size_t max_iter = p - shift;
  const double * const t_lam = lam + shift;
  double * const t_lam_star = lam_star + shift;

  // Handle main part
  #pragma omp parallel for schedule(dynamic) if(max_iter >= PARALLEL_THRESHOLD)
  for (size_t i = 0; i < max_iter; ++i) {

    // Accumulators
    double fi  = 0; // \tilde f_n(\lambda_i) - density
    double hfi = 0; // H_{f_n} (\lambda_i) - hilbert transform

    // Temp variables
    double x, abs_x, x2, x4;
    double hfi_part, log_term, linear_term;
    double denom_p1, denom_p2;

    // Compute the kernel estimation
    for (size_t j = 0; j < max_iter; ++j) {
      x = (t_lam[i] - t_lam[j]) / (t_lam[j] * h);
      abs_x = fabs(x);
      x2 = SQUARE(x);
      x4 = SQUARE(x2);

      // hfi
      if (fabs(abs_x - SQRT5) < eps) {
        hfi_part = (-3.0 / 10.0 / M_PI) * x;
      } else if (abs_x > 5.0) {
        hfi_part = (-1.0 / (M_PI * x)) * (1.0 + (1.0 / x2) + 15.0 / (7.0 * x4));
      } else {
        log_term = log(fabs((SQRT5 - x) / (SQRT5 + x)));
        linear_term = (-3.0 / 10.0 / M_PI) * x;
        hfi_part = linear_term + (c1 / M_PI) * (1 - x2 / 5.0) * log_term;
      }
      hfi += hfi_part / (h * t_lam[j]);

      // fi
      fi += c1 * relu(1 - SQUARE(x) / 5) / (t_lam[j] * h);
    }

    // Convert sums to means
    fi /= max_iter;
    hfi /= max_iter;

    if (p <= n) {
      denom_p1 = M_PI * ratio * fi * t_lam[i];
      denom_p2 = 1 - ratio - M_PI * ratio * hfi * t_lam[i];
      t_lam_star[i] = t_lam[i] / (SQUARE(denom_p1) + SQUARE(denom_p2));
    } else {
      denom_p1 = M_PI * M_PI * SQUARE(t_lam[i]);
      denom_p2 = SQUARE(fi) + SQUARE(hfi);
      t_lam_star[i] = t_lam[i] / (denom_p1 * denom_p2);
    }
  }

  // Handle the zero eigenvalues
  if (shift > 0) {

    double inv_lams_mean = 0;
    for (size_t i = 0; i < max_iter; ++i) {
      inv_lams_mean += 1 / t_lam[i];
    }
    inv_lams_mean /= max_iter;

    double hf0 = (
      (1.0 / M_PI) * (
        (3.0 / 10.0 / SQUARE(h)) + 
          (3.0 / 4.0 / SQRT5 / h) *
          (1.0 - 1.0 / 5.0 / SQUARE(h)) *
          log((1.0 + SQRT5 * h) / (1 - SQRT5 * h))
      )
    ) * inv_lams_mean;

    double dtilde0 = 1 / (M_PI * (d_p - d_n) / d_n * hf0);
    for (size_t i = 0; i < shift; ++i) {
      lam_star[i] =  dtilde0;
    }

  }

  return ;
}

void C_LWLinear(
    const double * const data, // Data n x p (c contiguous)
    double * const sample_cov_star, // Shrunk covariance buffer
    size_t n, // Number of samples used
    size_t p // Number of features
) {
  size_t p2 = SQUARE(p);
  size_t n2 = SQUARE(n);

  memset(sample_cov_star, (double) 0.0, p2 * sizeof(double));

  // Precompute data^2
  double * data2 = malloc(n * p * sizeof(double));
  #pragma omp parallel for if(n * p >= SQUARE(PARALLEL_THRESHOLD))
    for (size_t i = 0; i < n * p; ++i) {
        data2[i] = SQUARE(data[i]);
  }

  // Construct cov and compute beta_
  double beta_ = 0.0;
  // The order of pi k pj is chosen as to prevent write collisions to sample_cov_star
  #pragma omp parallel for reduction(+:beta_) if(p >= PARALLEL_THRESHOLD)
  for (size_t pi = 0; pi < p; ++pi) {
      double beta_part = 0.0;
      for (size_t k = 0; k < n; ++k) {
          double data_ik = data[k*p + pi];
          double x2_ik = data2[k*p + pi];
          for (size_t pj = 0; pj < p; ++pj) {
              sample_cov_star[pi*p + pj] += data_ik * data[k*p + pj];
              beta_part += x2_ik * data2[k*p + pj];
          }
      }
      beta_ += beta_part;
  }

  free(data2);

  // Compute delta_ from the cov
  // and scale the cov
  double delta_ = 0.0;
  #pragma omp parallel for reduction(+:delta_) if(p >= PARALLEL_THRESHOLD)
  for (size_t pi = 0; pi < p; ++pi) {
    double delta_part = 0.0;
    for (size_t pj = 0; pj < p; ++pj) {
      delta_part += SQUARE(sample_cov_star[pi*p + pj]);
      sample_cov_star[pi*p + pj] /= n;
    }
    delta_ += delta_part;
  }
  delta_ /= n2;

  double cov_trace = trace(sample_cov_star, p);
  double mu = cov_trace / p;

  double beta = 1.0 / (n * p) * (beta_ / n - delta_);
  double delta = delta_ - (SQUARE(mu) * p);
  delta /= p;
  beta = MIN(beta, delta);
  double shrinkage = beta < DOUBLE_EPS ? 1.0 : clip(beta / delta, 0, 1);

  // Shrink the cov
  scalar_multiply(sample_cov_star, p2, (1.0 - shrinkage));

  // Add on the diagonal
  double add_value = shrinkage * mu;
  for (size_t i = 0; i < p; ++i) {
    sample_cov_star[i*p + i] += add_value;
  }

  return ;
}
