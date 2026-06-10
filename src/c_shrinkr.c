#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "c_shrinkr.h"
#define SQRT5 2.23606797749979
#define PHI   0.61803398874985
#define PARALLEL_THRESHOLD_LWA 100
#define PARALLEL_THRESHOLD 200
#define PARALLEL_THRESHOLD_TRACE 40
#define DOUBLE_EPS 1e-12
#define SQUARE(x) ((x) * (x))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) > (y)) ? (y) : (x))
#define DEAL_MAX_ITERS 200
#define DEAL_EPS 1e-8

#ifdef _OPENMP
#include <omp.h>
#endif

// Utilities functions

inline double relu(double x) {
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

// Computes tr(S) for a square matrix S
double trace(const double * const S, size_t p) {
  double acc = 0;
  for (size_t i = 0; i < p; ++i) {
    acc += S[i*p + i];
  }
  return acc;
}

// Computes tr(S @ S.T) for a symmetric matrix S
double traceS2(const double * const S, size_t p) {
  double acc = 0;
  const size_t max_iter = SQUARE(p);
  #pragma omp parallel for reduction(+:acc) if(p >= PARALLEL_THRESHOLD_TRACE)
  for (size_t i = 0; i < max_iter ; ++i) {
    acc += SQUARE(S[i]);
  }
  return acc;
}


// Computes tr(A @ A.T), where A is the diagonal matrix of S
double traceDiagS2(const double * const S, size_t p) {
  double acc = 0;
  for (size_t i = 0; i < p ; ++i) {
    acc += SQUARE(S[i*p + i]);
  }
  return acc;
}


// Compute the sum of the || x_k ||_2^4 for x_k being the k-th sample (row)
double sumNorm2p4(const double * const data, size_t n, size_t p) {
  double sum_norms_4 = 0.0;
  #pragma omp parallel for reduction(+:sum_norms_4) if(n >= PARALLEL_THRESHOLD)
  for (size_t ni = 0; ni < n; ++ni) {
    double norm_sq = 0.0;
    for (size_t pi = 0; pi < p; ++pi) {
      const double val = data[ni*p + pi];
      norm_sq += SQUARE(val);
    }
    sum_norms_4 += SQUARE(norm_sq);
  }

  return sum_norms_4;
}


void scalar_multiply(double * const data, size_t n, double scale) {
  #pragma omp parallel for if(n >= PARALLEL_THRESHOLD)
  for (size_t i = 0; i < n ; ++i) {
    data[i] = scale * data[i];
  }
}

void scalar_multiply_copy(const double * const data, double * const output, size_t n, double scale) {
  #pragma omp parallel for if(n >= PARALLEL_THRESHOLD)
  for (size_t i = 0; i < n ; ++i) {
    output[i] = scale * data[i];
  }
}


// Main implementation ----------------------------------------------------------------------------


double C_OAS(
    const double * const sample_cov, // Sample covariance
    double * const sample_cov_star, // Shrunk covariance
    size_t n, // Number of samples used
    size_t p  // Dimensions of the sample_cov
) {
  size_t p2 = SQUARE(p);

  const double alpha = traceS2(sample_cov, p) / p2;
  const double mu = trace(sample_cov, p) / p;
  const double mu_squared = SQUARE(mu);

  const double num = alpha + mu_squared;
  const double denom = (n + 1) * (alpha - mu_squared / p);
  const double shrinkage = denom < DOUBLE_EPS ? 1.0 : clip(num / denom, 0, 1);

  // Shrink the cov
  const double scale = 1.0 - shrinkage;
  #pragma omp parallel for if(p2 >= PARALLEL_THRESHOLD)
  for (size_t i = 0; i < p2; ++i) {
    sample_cov_star[i] = sample_cov[i] * scale;
  }

  // Add on the diagonal
  const double add_value = shrinkage * mu;
  for (size_t i = 0; i < p; ++i) {
    sample_cov_star[i*p + i] += add_value;
  }

  return shrinkage;
}


void C_LWAnalytical(
    const double * const lam, // Input data
    double * const lam_star, // Output data
    size_t n, // Number of sample used
    size_t p, // Dimensions of both x and y
    double eps // Epsilon value
) {
  const double h = pow(n, - 1.0 / 3.0);
  const double d_p = (double) p;
  const double d_n = (double) n;
  const double ratio = d_p / d_n;
  const double c1 = 3.0 / (4.0 * SQRT5); // \frac{3}{4\sqrt{5}}
  const double c2 = c1 / M_PI; // \frac{3}{4\sqrt{5}}
  const double c_linear = (-3.0 / 10.0 / M_PI); // \frac{-3}{10 \pi}

  size_t shift = n < p ? p - n : 0;
  size_t max_iter = p - shift;
  const double * const t_lam = lam + shift;
  double * const t_lam_star = lam_star + shift;

  double* inv_lam_h = malloc(max_iter * sizeof(double));
  for (size_t i = 0; i < max_iter; ++i) {
      inv_lam_h[i] = 1.0 / (t_lam[i] * h);
  }

  // Handle main part
  #pragma omp parallel for schedule(guided) if(max_iter >= PARALLEL_THRESHOLD_LWA)
  for (size_t i = 0; i < max_iter; ++i) {

    // Accumulators
    double fi  = 0.0; // \tilde f_n(\lambda_i) - density
    double hfi = 0.0; // H_{f_n} (\lambda_i) - hilbert transform

    // Temp variables
    double hfi_part, log_term;

    // Compute the kernel estimation
    const double lam_i = t_lam[i];
    for (size_t j = 0; j < max_iter; ++j) {
      const double x = (lam_i - t_lam[j]) * inv_lam_h[j];
      const double abs_x = fabs(x);
      const double x2 = SQUARE(x);

      if (abs_x > 5.0) {
        const double inv_x2 = 1 / x2;
        const double inv_x4 = SQUARE(inv_x2);
        hfi_part = ((-1.0 / M_PI) / x) * (1.0 + inv_x2 + (15.0 / 7.0) * inv_x4);
      } else {
        const double linear_term = c_linear * x;
        if (fabs(abs_x - SQRT5) > eps) {
          log_term = c2 * (1 - (x2 / 5.0)) * log(fabs((SQRT5 - x) / (SQRT5 + x)));
        } else {
          log_term = 0.0;
        }
        hfi_part = linear_term + log_term;
      }
      hfi += hfi_part * inv_lam_h[j];

      // fi
      fi += c1 * relu(1.0 - SQUARE(x) / 5) * inv_lam_h[j];
    }

    // Convert sums to means
    fi /= max_iter;
    hfi /= max_iter;

    if (p <= n) {
      const double denom_p1 = M_PI * ratio * fi * t_lam[i];
      const double denom_p2 = 1.0 - ratio - M_PI * ratio * hfi * t_lam[i];
      t_lam_star[i] = t_lam[i] / (SQUARE(denom_p1) + SQUARE(denom_p2));
    } else {
      const double denom_p1 = M_PI * M_PI * SQUARE(t_lam[i]);
      const double denom_p2 = SQUARE(fi) + SQUARE(hfi);
      t_lam_star[i] = t_lam[i] / (denom_p1 * denom_p2);
    }
  }
  free(inv_lam_h);

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

double C_LWLinear(
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
          const double data_ik = data[k*p + pi];
          const double data2_ik = data2[k*p + pi];
          for (size_t pj = 0; pj < p; ++pj) {
              sample_cov_star[pi*p + pj] += data_ik * data[k*p + pj];
              beta_part += data2_ik * data2[k*p + pj];
          }
      }
      beta_ += beta_part;
  }

  free(data2);

  // Compute delta_ from the cov
  // and scale the cov
  double delta_ = 0.0;
  const double inv_n = 1.0 / n;
  #pragma omp parallel for reduction(+:delta_) if(p >= PARALLEL_THRESHOLD)
  for (size_t pi = 0; pi < p; ++pi) {
    double delta_part = 0.0;
    for (size_t pj = 0; pj < p; ++pj) {
      delta_part += SQUARE(sample_cov_star[pi*p + pj]);
      sample_cov_star[pi*p + pj] *= inv_n;
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

  return shrinkage;
}


double C_LWLinearFast(
    const double * const data, // Data n x p (c contiguous)
    const double * const sample_cov, // Sample covariance (p x p) (c contiguous)
    double * const sample_cov_star, // Shrunk covariance buffer
    size_t n, // Number of samples used
    size_t p // Number of features
) {
  size_t p2 = SQUARE(p);
  size_t n2 = SQUARE(n);

  // Compute mu
  double cov_trace = trace(sample_cov, p);
  double mu = cov_trace / p;

  // Compute delta and beta
  double delta_ = traceS2(sample_cov, p);
  double beta_ = sumNorm2p4(data, n, p);
  double beta = 1.0 / (p * n) * (beta_ / n - delta_);
  double delta = delta_ - 2.0 * mu * cov_trace + (SQUARE(mu) * p);
  delta /= p;
  beta = MIN(beta, delta);

  double shrinkage = beta < DOUBLE_EPS ? 1.0 : clip(beta / delta, 0, 1);

  // Shrink the cov
  scalar_multiply_copy(sample_cov, sample_cov_star, p2, (1.0 - shrinkage));

  // Add on the diagonal
  double add_value = shrinkage * mu;
  for (size_t i = 0; i < p; ++i) {
    sample_cov_star[i*p + i] += add_value;
  }

  return shrinkage;
}


double C_DEALObjective(
    const double * const base_evals,
    const double * const surr_evals,
    const double * const z_vec,
    double gamma,
    double * start_value,
    size_t n,
    size_t p
) {
  double * const inv_diag = malloc(p * sizeof(double));

  // Compute delta
  double delta = *start_value;
  double curr_delta;

  for (size_t ii = 0; ii < DEAL_MAX_ITERS; ++ii) {
    curr_delta = 0.0;
    for (size_t i = 0; i < p; ++i) {
      inv_diag[i] = gamma + (surr_evals[i] / (1 + delta));
      curr_delta += surr_evals[i] / inv_diag[i];
    }
    curr_delta /= n;
    if (fabs(delta - curr_delta) <= DEAL_EPS) {
      break;
    }
    delta = curr_delta;
  }

  // Set the new start_value (for the future)
  *start_value = delta;

  // Compute delta prime
  double a = 0.0;
  double b = 0.0;
  double inner;
  for (size_t i = 0; i < p; ++i) {
    inner = surr_evals[i] / inv_diag[i];
    a += surr_evals[i] / SQUARE(inv_diag[i]);
    b += SQUARE(inner);
  }
  b /= SQUARE(1 + delta);
  double delta_prime = (-a) / (n - b);

  // Compute a_gamma, b_gamma
  double a_gamma = 0.0;
  double b_gamma = 0.0;
  double z2_i, beta_i, diag_i, delta_inv_diagT_i;
  for (size_t i = 0; i < p; ++i) {
    z2_i = SQUARE(z_vec[i]);
    beta_i = 1.0 / (1.0 + delta);
    diag_i = 1.0 / inv_diag[i];
    delta_inv_diagT_i = 1 - delta_prime * surr_evals[i] * SQUARE(beta_i);

    b_gamma += z2_i * diag_i;
    a_gamma -= z2_i * base_evals[i] * SQUARE(diag_i) * delta_inv_diagT_i;
  }

  free(inv_diag);

  return SQUARE(b_gamma) / a_gamma;
}


double C_DEAL(
    const double * const base_evals,
    const double * const surr_evals,
    const double * const z_vec,
    double gamma_min,
    double gamma_max,
    size_t n,
    size_t p
) {
  double delta = 1.0;

  #define cost(log_gamma) C_DEALObjective(base_evals, surr_evals, z_vec, exp(log_gamma), &delta, n, p)

  // a, b, c and d are the points for the golden section search
  double a, b, c, d;
  double cost_c, cost_d;

  a = log(gamma_min);
  b = log(gamma_max);
  do {
    c = b - PHI * (b - a);
    d = a + PHI * (b - a);
    cost_c = cost(c);
    cost_d = cost(d);

    if (cost_c < cost_d) {
      b = d;
      d = c;
      cost_d = cost_c;
      c = b - PHI * (b - a);
      cost_c = cost(c);
    } else {
      a = c;
      c = d;
      cost_c = cost_d;
      d = a + PHI * (b - a);
      cost_d = cost(d);
    }
  } while (fabs(a - b) > DEAL_EPS);

  double optimal_log_gamma = (a + b) / 2.0;
  double optimal_gamma = exp(optimal_log_gamma);
  return optimal_gamma;
}
