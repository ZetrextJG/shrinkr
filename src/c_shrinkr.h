#ifndef LIBSHRINK_H
#define LIBSHRINK_H

#include "stddef.h"

void C_OAS(
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

void C_LWLinear(
    const double * const data, // Data n x p
    double * const sample_cov_star, // Shrunk covariance buffer
    size_t n, // Number of samples used
    size_t p, // Dimensions of both x and y
    size_t block_size // Block size used for computation
);

#endif
