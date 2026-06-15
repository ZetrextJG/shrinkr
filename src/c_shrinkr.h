#ifndef LIBSHRINK_H
#define LIBSHRINK_H

#include "stddef.h"

/**
 * Oracle Approximating Shrinkage (OAS) covariance estimator.
 *
 * The formulation is based on https://arxiv.org/pdf/0907.4698.pdf.
 *
 * @param sample_cov Sample covariance matrix (pxp) (C or F contiguous)
 * @param sample_cov_star Output buffer for the shurnk covariance (pxp) (C or F contiguous but matching sample_cov)
 * @param n Number of samples used to compute the sample covariance
 * @param p One of the dimensions of the matrix
 *
 * @returns The value of the linear shrinkage
 *
 * @ingroup public_api
 */
double C_OAS(
    const double * const sample_cov, // Sample covariance
    double * const sample_cov_star, // Shrunk covariance buffer
    size_t n, // Number of samples used
    size_t p  // Dimensions of the sample_cov
);

/**
 * Ledoit-Wolf Analytical (nonlinear) shrinkage of eigenvalues.
 *
 * Based on Ledoit and Wolf (2018), using the analytic formula that avoids
 * numerical optimization. Handles the high-dimensional setting where p > n.
 *
 * @param lam Array of length p containing eigenvalues of the sample covariance matrix
 * @param lam_star Array buffer of length p for shrunk eigenvalues
 * @param sample_cov_star Output buffer for the shurnk covariance (pxp) (C or F contiguous but matching sample_cov)
 * @param n Number of samples used to compute the sample covariance
 * @param p Number of variables. Length of lam and lam_star
 * @param eps Epsilon value for numerical stability
 *
 * @ingroup public_api
 */
void C_LWAnalytical(
    const double * const lam, // Input data
    double * const lam_star, // Output data
    size_t n, // Number of samples used
    size_t p, // Dimensions of both x and y
    double eps // Epsilon value
);

/**
 * Ledoit-Wolf linear shrinkage estimator.
 *
 * Based on the Ledoit-Wolf Lemma. http://www.ledoit.net/ole1a.pdf
 *
 * @param data Data matrix (nxp) (C contiguous)
 * @param sample_cov Sample covariance matrix (pxp) (C or F contiguous)
 * @param sample_cov_star Output buffer for the shurnk covariance (pxp) (C or F contiguous but matching sample_cov)
 * @param n Number of samples used to compute the sample covariance
 * @param p One of the dimensions of the covariance matrix
 *
 * @ingroup public_api
 */
double C_LWLinear(
    const double * const data, // Data n x p
    const double * const sample_cov, // Sample covariance
    double * const sample_cov_star, // Shrunk covariance buffer
    size_t n, // Number of samples used
    size_t p // Dimensions of both x and y
);

/**
 * Objective function of DEAL.
 *
 * For more information check ``shrinkr.functional.deal``.
 *
 * @param base_evals Array of length p of eigenvalues for the objective. Which will be shrunk.
 * @param surr_evals Array of length p of eigenvalues for the objective. Used to compute shrinkage paramters.
 * @param z_vec Array of length p of the vector of interest projected into the eigenvector space.
 * @param gamma The value of gamma to evaluate. During optimization only this value changes.
 * @param start_value Starting value of delta for the fixed point iteration method used by for the objective. 
 * @param n Number of effective samples used to compute the sample covariance
 * @param p One of the dimensions of the covariance matrix. Number of features.
 *
 * @ingroup public_api
 */
double C_DEALObjective(
    const double * const base_evals,
    const double * const surr_evals,
    const double * const z_vec,
    double gamma,
    double * start_value,
    size_t n,
    size_t p
);

/**
 * DEAL (Deterministic Equivalents for Adaptive LDA) shrinkage.
 *
 * For more information check ``shrinkr.functional.deal``.
 *
 * @param base_evals Array of length p of eigenvalues for the objective. Which will be shrunk.
 * @param surr_evals Array of length p of eigenvalues for the objective. Used to compute shrinkage paramters.
 * @param z_vec Array of length p of the vector of interest projected into the eigenvector space.
 * @param gamma_min Minimum value for the gamma bounded search.
 * @param gamma_max Maximum value for the gamma bounded search.
 * @param n Number of effective samples used to compute the sample covariance
 * @param p One of the dimensions of the covariance matrix. Number of features.
 *
 * @ingroup public_api
 */
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
