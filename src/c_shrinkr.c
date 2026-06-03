#include <stdio.h>
#include <math.h>
#include "c_shrinkr.h"
#define SQRT5 2.23606797749979
#define PARALLEL_THRESHOLD 200

#ifdef _OPENMP
#include <omp.h>
#endif

double square(double x) {
  return x * x;
}

double relu(double x) {
  return x > 0.0 ? x : 0.0;
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
      x2 = square(x);
      x4 = square(x2);

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
      fi += c1 * relu(1 - square(x) / 5) / (t_lam[j] * h);
    }

    // Convert sums to means
    fi /= max_iter;
    hfi /= max_iter;

    if (p <= n) {
      denom_p1 = M_PI * ratio * fi * t_lam[i];
      denom_p2 = 1 - ratio - M_PI * ratio * hfi * t_lam[i];
      t_lam_star[i] = t_lam[i] / (square(denom_p1) + square(denom_p2));
    } else {
      denom_p1 = M_PI * M_PI * square(t_lam[i]);
      denom_p2 = square(fi) + square(hfi);
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
        (3.0 / 10.0 / square(h)) + 
          (3.0 / 4.0 / SQRT5 / h) *
          (1.0 - 1.0 / 5.0 / square(h)) *
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
