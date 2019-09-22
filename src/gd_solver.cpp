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
    m_x = randn(1, m_L.n_rows);
    mat objective_prev = {pow(10, 16)};
    m_s = s;
    mat gradient, s_estimate, rel_obj_change;

    for (size_t n = 0; n < m_max_iter; n++)
    {

        // Calculate current value of objective
        s_estimate = f_model(m_x, m_L);
        m_objective = f_objective(s_estimate, s);

        // Calculate numerical gradient at point x
        gradient = calculate_gradient();

        // Update solution using gradient decent
        m_x = m_x - m_lr * m_objective * gradient;

        // Check termination condition
        rel_obj_change = (objective_prev - m_objective) / m_objective;
        if (as_scalar(rel_obj_change) < m_termination_threshold)
        {
            cout << "Change in objective value smaller than specified threshold" << endl;
            cout << "Iteration terminated at round " << n << endl;
            return m_x;
        }
        objective_prev = m_objective;
    }

    cout << "Maximum number of iterations was reached" << endl;
    return m_x;
}

arma::mat GDSolver::calculate_gradient()
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
    return gradient;
}