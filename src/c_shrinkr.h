#ifndef LIBSHRINK_H
#define LIBSHRINK_H

#include "stddef.h"


void C_LWAnalytical(
    const double * const lam, // Input data
    double * const lam_star, // Output data
    size_t n, // Number of sample used
    size_t p, // Dimensions of both x and y
    double eps // Epsilon value
);

#endif
