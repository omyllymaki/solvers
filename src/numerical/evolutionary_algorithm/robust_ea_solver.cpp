#include "robust_ea_solver.h"
#include "../../common.h"

// Trimmed mean of absolute errors
// Rejects certain proportion of largest residuals
// Should be robust against some amount of outlier signal values
arma::mat RobustEASolver::objective(arma::mat estimate, arma::mat expected)
{
    return trimmed_mae(estimate.t(), expected.t(), m_rejection_threshold); 
}

void RobustEASolver::set_rejection_threshold(double value) {
    m_rejection_threshold = value;
}
