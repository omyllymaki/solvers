#include "ea_solver.h"
#include "../../common.h"
#include "../../logging/easylogging++.h"

using namespace std;

EASolver::EASolver(const arma::mat &L,
                   const arma::mat stdev_scaling_factors,
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

    if (stdev_scaling_factors.n_cols != m_L.n_rows)
    {
        throw std::invalid_argument("Invalid stdev_scaling_factors dimension.");
    }
    else
    {
        m_stdev_scaling_factors = stdev_scaling_factors;
    }
}

// Overload constructor
// double stdev_scaling_factor will be converted to arma::matrix
EASolver::EASolver(const arma::mat &L,
                   const int n_candidates,
                   const int max_iter,
                   const double objective_threshold,
                   const int n_no_change_threshold,
                   double stdev_scaling_factor) : EASolver(L,
                                                           stdev_scaling_factor * arma::ones(1, L.n_rows),
                                                           n_candidates, max_iter,
                                                           objective_threshold,
                                                           n_no_change_threshold){};

arma::mat EASolver::objective(arma::mat estimate, arma::mat expected)
{
    return rmse(estimate, expected);
}

// Generate candidate solutions
// Test candidate solutions using model and objective functions
// Take the best candidate and generate new candidates around that
// Terminate when any of the termination criteria is fulfilled
arma::mat EASolver::solve(const arma::mat &s)
{
    m_s = s;

    initialize();
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
        LOG(INFO) << "Objective values smaller than specified threshold";
        LOG(INFO) << "Iteration terminated at round " << m_round;
        return true;
    }

    if (m_no_change_counter > m_n_no_change_threshold)
    {
        LOG(INFO) << "Objective value didn't change for last " << m_n_no_change_threshold << " rounds";
        LOG(INFO) << "Iteration terminated at round " << m_round;
        return true;
    }

    return false;
}

// In every iteration, we can update stdevs which are used to generate new population
// Usually, we want large stdevs at first to ensure that we find global minimum instead of local one
// During iteration, we want to deacrease stdevs in order to find minimum more accurately
void EASolver::update_stdevs()
{
    // Dynamic factor changes smoothly from 1 to 0 during iteration
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
    // Best candidate is the one which has lowest objective value
    m_min_index = m_obj_values.index_min();
    m_best_obj_value = m_obj_values.row(m_min_index);
    m_x = m_candidates.row(m_min_index);
}

void EASolver::set_initial_guess(arma::mat initial_guess)
{
    m_init_guess = initial_guess;
}

void EASolver::initialize()
{
    if (m_init_guess.is_empty())
    {
        m_x = arma::zeros(1, m_L.n_rows);
    }
    else
    {
        m_x = m_init_guess;
    }

    arma::mat s_estimate = model(m_x, m_L);
    m_best_obj_value = objective(s_estimate, m_s);
    int m_no_change_counter = 0;
}