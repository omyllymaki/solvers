#include "gd_solver.h"
#include "../common.h"
#include "../logging/easylogging++.h"
#include <iostream>

using arma::mat;
using arma::randn;

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
            LOG(INFO) << "Change in objective value smaller than specified threshold";
            LOG(INFO) << "Iteration terminated at round " << m_round;
            return m_x;
        }

        m_objective_prev = m_objective;
    }

    LOG(INFO) << "Maximum number of iterations was reached";
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

void GDSolver::set_learning_rate(double lr)
{
    m_lr = lr;
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

arma::mat GDSolver::get_objective_value()
{
    return m_objective;
}

double GDSolver::find_optimal_lr(const arma::mat &s, arma::mat lr_array, int n_iter)
{
    arma::mat objective_values;
    for (size_t i = 0; i < lr_array.n_elem; i++)
    {
        auto solver = GDSolver(m_L, lr_array[i], n_iter);
        arma::mat result = solver.solve(s);
        arma::mat obj = solver.get_objective_value();
        LOG(DEBUG) << "lr " << lr_array[i] << ": objective " << obj;
        objective_values.insert_cols(i, obj);
    }

    int min_index = objective_values.index_min();
    double best_lr = lr_array[min_index];
    LOG(INFO) << "Optimal learning rate found: " << best_lr;
    m_lr = best_lr;

    return best_lr;
}

double GDSolver::find_optimal_lr(const arma::mat &s, double lb, double ub, int n_candidates, int n_iter, std::string method)
{
    arma::mat lr_array;
    if (method == "lin")
    {
        lr_array = arma::linspace(lb, ub, n_candidates);
    }
    else if (method == "log")
    {
        lr_array = arma::logspace(lb, ub, n_candidates);
    }
    else
    {
        throw std::invalid_argument("Invalid method");
    }
    LOG(DEBUG) << "Learning rate candidates: " << lr_array;
    double best_lr = find_optimal_lr(s, lr_array, n_iter);
    return best_lr;
}