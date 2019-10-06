#include "gn_solver.h"
#include "../common.h"
#include <iostream>

using arma::mat;
using arma::randn;
using std::cout;
using std::endl;

GNSolver::GNSolver(const arma::mat &L,
                   const int max_iter,
                   const double termination_threshold)
{
    m_L = L;
    m_max_iter = max_iter;
    m_termination_threshold = termination_threshold;
}

arma::mat GNSolver::solve(const arma::mat &s)
{
    m_objective_prev = {std::pow(10, 16)};
    m_x = randn(1, m_L.n_rows);
    m_s = s;

    arma::mat gradient, inv_gradient, s_estimate, x, derivative, residual, residual0;

    for (m_round = 0; m_round < m_max_iter; m_round++)
    {

       
        s_estimate = model(m_x, m_L);
        residual0 = s_estimate - m_s;

        gradient.reset();
        for (size_t i = 0; i < m_x.n_elem; i++)
        {
            x = m_x;
            x[i] = x[i] + m_x_delta;
            s_estimate = model(x, m_L);
            residual = s_estimate - m_s;
            derivative = (residual - residual0) / m_x_delta;
            gradient.insert_rows(i, derivative);
        }
        inv_gradient = calculate_svd_inverse(gradient.t());
        arma::mat step = inv_gradient * residual0.t();

        m_x = m_x - step.t();

        // cout << m_x << endl;

        m_objective_prev = m_objective;
    }

    return m_x;

}

arma::mat GNSolver::model(mat x, mat L)
{
    return x * L;
}
