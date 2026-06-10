#ifndef LIBSHRINK_H
#define LIBSHRINK_H

#include "stddef.h"

double C_OAS(
    const double * const sample_cov, // Sample covariance
    double * const sample_cov_star, // Shrunk covariance buffer
    size_t n, // Number of samples used
    size_t p  // Dimensions of the sample_cov
);

void C_LWAnalytical(
    const double * const lam, // Input data
    double * const lam_star, // Output data
    size_t n, // Number of samples used
    size_t p, // Dimensions of both x and y
    double eps // Epsilon value
);

double C_LWLinear(
    const double * const data, // Data n x p
    double * const sample_cov_star, // Shrunk covariance buffer
    size_t n, // Number of samples used
    size_t p // Dimensions of both x and y
);

double C_DEALObjective(
    const double * const base_evals,
    const double * const surr_evals,
    const double * const z_vec,
    double gamma,
    double * start_value,
    size_t n,
    size_t p
);

double C_DEAL(
    const double * const base_evals,
    const double * const surr_evals,
    const double * const z_vec,
    double gamma_min,
    double gamma_max,
    size_t n,
    size_t p
);

#endif
