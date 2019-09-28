#include "solver.h"

arma::mat Solver::get_signal_residual() {
    return get_signal_estimate() - m_s;
}
