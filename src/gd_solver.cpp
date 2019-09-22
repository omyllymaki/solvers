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
    const double x_delta = pow(10, -6);
    mat x = randn(1, m_L.n_rows);
    mat objective_prev = {pow(10, 16)};

    for (size_t n = 0; n < m_max_iter; n++)
    {

        // Calculate current value of objective
        mat s_estimate = f_model(x, m_L);
        mat objective0 = f_objective(s_estimate, s);

        // Calculate numerical gradient at point x
        mat gradient;
        for (size_t i = 0; i < x.n_elem; i++)
        {
            mat xt = x;
            xt[i] = x[i] + x_delta;
            s_estimate = f_model(xt, m_L);

            mat objective = f_objective(s_estimate, s);

            mat derivative = (objective - objective0) / x_delta;
            gradient.insert_cols(i, derivative);
        }

        // Update solution using gradient decent
        mat x_step = m_lr * objective0 * gradient;
        x = x - x_step;

        // Check termination condition
        mat rel_obj_change = (objective_prev - objective0) / objective0;
        if (as_scalar(rel_obj_change) < m_termination_threshold)
        {
            cout << "Change in objective value smaller than specified threshold" << endl;
            cout << "Iteration terminated at round " << n << endl;
            return x;
        }
        objective_prev = objective0;
    }

    cout << "Maximum number of iterations was reached" << endl;
    return x;
}