#include "robust_ea_solver.h"

// Trimmed mean of absolute errors
// Rejects certain proportion of largest residuals
// Should be robust against some amount of outlier signal values
arma::mat RobustEASolver::objective(arma::mat estimate, arma::mat expected)
{
    arma::mat residual = estimate - expected;
    arma::mat abs_residual = arma::abs(residual);
    int n_points = round(m_rejection_threshold * abs_residual.n_elem);
    arma::vec abs_residual_sorted = arma::sort(abs_residual.t(), "descent");
    return arma::mean(abs_residual_sorted.rows(n_points, abs_residual_sorted.n_elem - 1), 0);  
}

void RobustEASolver::set_rejection_threshold(double value) {
    m_rejection_threshold = value;
}
