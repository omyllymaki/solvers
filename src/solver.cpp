#include "solver.h"

arma::mat Solver::get_signal_residual() {
    return get_signal_estimate() - m_s;
}

arma::mat Solver::get_signal_estimate() {
    return model(m_x, m_L);
}