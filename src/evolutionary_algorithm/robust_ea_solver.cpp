#include "robust_ea_solver.h"
#include "../common.h"

// Trimmed mean of absolute errors
// Should be robust against some outlier values in signal
arma::mat RobustEASolver::objective(arma::mat estimate, arma::mat expected)
{
    arma::mat residual = estimate - expected;
    arma::mat abs_residual = arma::abs(residual);
    return trimmed_mean(abs_residual.t(), 0.05f); // Rejects 10 % of the values
}