#include "gd_solver.h"
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

    for (size_t n = 0; n < m_max_iter; n++)
    {
        s_estimate = f_model(m_x, m_L);
        m_objective = f_objective(s_estimate, s);
        update_gradient();
        update_solution();

        if (is_termination_condition_filled())
        {
            cout << "Change in objective value smaller than specified threshold" << endl;
            cout << "Iteration terminated at round " << n << endl;
            return m_x;
        }

        m_objective_prev = m_objective;
    }

    cout << "Maximum number of iterations was reached" << endl;
    return m_x;
}

void GDSolver::update_gradient()
{
    mat gradient, s_estimate, objective, x, derivative;
    for (size_t i = 0; i < m_x.n_elem; i++)
    {
        x = m_x;
        x[i] = x[i] + m_x_delta;
        s_estimate = f_model(x, m_L);
        objective = f_objective(s_estimate, m_s);
        derivative = (objective - m_objective) / m_x_delta;
        gradient.insert_cols(i, derivative);
    }
    m_gradient = gradient;
    ;
}

void GDSolver::update_solution()
{
    m_x = m_x - m_lr * m_objective * m_gradient;
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

mat GDSolver::f_objective(mat estimate, mat expected)
{
    mat residual = estimate - expected;
    return sqrt(sum(pow(residual, 2), 1) / residual.n_elem);
}

mat GDSolver::f_model(mat x, mat L)
{
    return x * L;
}