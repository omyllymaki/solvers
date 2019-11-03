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

arma::mat Solver::linear_model(arma::mat x, arma::mat L)
{
    return x * L;
}

void Solver::set_library(arma::mat L)
{
    m_L = L;
}

arma::mat Solver::get_library()
{
    return m_L;
}

void Solver::set_signal(arma::mat s)
{
    m_s = s;
}

model_wrapper Solver::get_model()
{
    return model;
}