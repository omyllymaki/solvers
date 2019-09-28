#include "solver.h"

arma::mat Solver::get_signal_residual()
{
    return get_signal_estimate() - m_s;
}

arma::mat Solver::get_signal_estimate()
{
    return model(m_x, m_L);
}

std::vector<arma::mat> Solver::solve_multiple(const std::vector<arma::mat> &signals)
{
    std::vector<arma::mat> solutions;
    for (auto &&signal : signals)
    {
        solutions.push_back(solve(signal));
    }
    return solutions;
}