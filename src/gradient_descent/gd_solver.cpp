#include "gd_solver.h"
#include "../common.h"
#include <iostream>

using arma::mat;
using arma::randn;
using std::cout;
using std::endl;

GDSolver::GDSolver(const arma::mat &L,
                   const double lr,
                   const int max_iter,
                   const double termination_threshold)
{
    m_L = L;
    m_lr = lr;
    m_max_iter = max_iter;
    m_termination_threshold = termination_threshold;
}

arma::mat GDSolver::solve(const arma::mat &s)
{
    m_objective_prev = {std::pow(10, 16)};
    m_x = randn(1, m_L.n_rows);
    m_s = s;
    mat s_estimate;

    for (m_round = 0; m_round < m_max_iter; m_round++)
    {
        s_estimate = model(m_x, m_L);
        m_objective = objective(s_estimate, s);
        update_gradient();
        update_solution();
        update_learning_rate();

        if (is_termination_condition_filled())
        {
            cout << "Change in objective value smaller than specified threshold" << endl;
            cout << "Iteration terminated at round " << m_round << endl;
            return m_x;
        }

        m_objective_prev = m_objective;
    }

    cout << "Maximum number of iterations was reached" << endl;
    return m_x;
}

void GDSolver::update_gradient()
{
    arma::mat gradient, s_estimate, x, derivative;
    for (size_t i = 0; i < m_x.n_elem; i++)
    {
        x = m_x;
        x[i] = x[i] + m_x_delta;
        s_estimate = model(x, m_L);
        derivative = (objective(s_estimate, m_s) - m_objective) / m_x_delta;
        gradient.insert_cols(i, derivative);
    }
    m_gradient = gradient;
    ;
}

void GDSolver::update_solution()
{
    m_x = m_x - m_lr * m_objective * m_gradient;
}

void GDSolver::update_learning_rate()
{
    m_lr = m_lr;
}

bool GDSolver::is_termination_condition_filled()
{
    mat rel_obj_change = (m_objective_prev - m_objective) / m_objective;
    if (as_scalar(rel_obj_change) < m_termination_threshold)
    {
        return true;
    }
    else
    {
        return false;
    }
}

arma::mat GDSolver::objective(arma::mat estimate, arma::mat expected)
{
    return rmse(estimate, expected);
}

arma::mat GDSolver::model(mat x, mat L)
{
    return x * L;
}