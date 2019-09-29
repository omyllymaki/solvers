#include "ea_solver.h"

using namespace std;

EASolver::EASolver(const arma::mat &L,
                   const int n_candidates,
                   const int max_iter,
                   const double objective_threshold,
                   const int n_no_change_threshold)
{
    m_L = L;
    m_n_candidates = n_candidates;
    m_max_iter = max_iter;
    m_objective_threshold = objective_threshold;
    m_n_no_change_threshold = n_no_change_threshold;
}

arma::mat EASolver::model(arma::mat x, arma::mat L)
{
    return x * L;
}

arma::mat EASolver::objective(arma::mat estimate, arma::mat expected)
{
    arma::mat residual = estimate - expected;
    return sqrt(sum(pow(residual, 2), 1) / residual.n_elem);
}

arma::mat EASolver::solve(const arma::mat &s)
{
    m_s = s;

    // TODO: add this as arguments
    m_stdev_scaling_factors = arma::ones(1, m_L.n_rows);
    m_x = arma::zeros(1, m_L.n_rows);

    arma::mat s_estimate = model(m_x, m_L);
    m_best_obj_value = objective(s_estimate, s);
    int m_no_change_counter = 0;

    for (m_round = 0; m_round < m_max_iter; m_round++)
    {
        update_stdevs();
        generate_and_test_candidates();
        update_solution();

        if (is_termination_condition_filled())
        {   
            return m_x;
        }
    }

    return m_x;
}

bool EASolver::is_termination_condition_filled()
{
    if (m_min_index == 0)
    {
        m_no_change_counter += 1;
    }
    else
    {
        m_no_change_counter = 0;
    }

    if (arma::as_scalar(m_best_obj_value) < m_objective_threshold)
    {
        cout << "Objective values smaller than specified threshold" << endl;
        cout << "Iteration terminated at round " << m_round << endl;
        return true;
    }

    if (m_no_change_counter > m_n_no_change_threshold)
    {
        cout << "Objective value didn't change for last " << m_n_no_change_threshold << " rounds" << endl;
        cout << "Iteration terminated at round " << m_round << endl;
        return true;
    }

    return false;
}

void EASolver::update_stdevs()
{
    double dynamic_factor = cos(m_round * M_PI / (2 * m_max_iter));
    m_stdevs = m_stdev_scaling_factors % arma::randn(1, m_L.n_rows) * dynamic_factor;
}

void EASolver::generate_and_test_candidates()
{
    m_candidates.reset();
    m_obj_values.reset();
    m_candidates.insert_rows(0, m_x);
    m_obj_values.insert_rows(0, m_best_obj_value);

    arma::mat s_estimate, obj_value, candidate;
    for (size_t i = 1; i < m_n_candidates; i++)
    {
        candidate = m_x + m_stdevs * arma::randn();
        s_estimate = model(candidate, m_L);
        obj_value = objective(s_estimate, m_s);
        m_obj_values.insert_rows(i, obj_value);
        m_candidates.insert_rows(i, candidate);
    }
}

void EASolver::update_solution()
{
    m_min_index = m_obj_values.index_min();
    m_best_obj_value = m_obj_values.row(m_min_index);
    m_x = m_candidates.row(m_min_index);
    // cout << m_min_index << " " << m_best_obj_value << endl;
}