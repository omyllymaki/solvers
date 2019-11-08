#include "ransac_solver.h"
#include "../common.h"
#include "../logging/easylogging++.h"

RansacSolver::RansacSolver(std::shared_ptr<Solver> solver,
                           int n_channels,
                           float accepted_error,
                           int n_accepted_points,
                           float objective_value_threshold,
                           int max_iter)
{
    m_solver = solver;
    m_n_channels = n_channels;
    m_n_max_iter = max_iter;
    m_accepted_error = accepted_error;
    m_n_accepted_points = n_accepted_points;
    m_objective_value_threshold = objective_value_threshold;
    model = m_solver->get_model();
}

RansacSolver::~RansacSolver()
{
}

arma::mat RansacSolver::solve_with_channel_subset(arma::uvec indices)
{
    arma::mat s_subset = m_s.elem(indices).t();
    arma::mat L_subset = m_L.cols(indices);
    m_solver->set_library(L_subset);
    arma::mat result = m_solver->solve(s_subset);
    return result;
}

arma::mat RansacSolver::objective(arma::mat residual)
{
    return sqrt(sum(pow(residual, 2), 1) / residual.n_elem);
}

arma::mat RansacSolver::solve(const arma::mat &s)
{
    m_s = s;
    m_L = m_solver->get_library();
    double lowest_objective_value = 100000000;

    for (size_t round = 0; round < m_n_max_iter; round++)
    {

        LOG(DEBUG) << "Round " << round;

        // Take n_channels channels randomly
        // Solve using just those channels
        std::vector<int> indices = sample_without_replacement(0, s.n_elem - 1, m_n_channels);
        arma::uvec indices_arma = arma::conv_to<arma::uvec>::from(indices);
        auto result = solve_with_channel_subset(indices_arma);
        LOG(DEBUG) << "Analysis result with random channels: " << result;

        // Using all channels, estimate assumed inliers
        // Assumed inlier is channel where residual is small enough
        m_solver->set_library(m_L);
        m_solver->set_signal(s);
        auto residual = m_solver->get_signal_residual();
        arma::uvec inlier_indices = arma::find(arma::abs(residual) < m_accepted_error);
        LOG(DEBUG) << "Number on inliers " << inlier_indices.size();

        // If number of assumed inliers is large enough, continue to evaluation
        if (inlier_indices.size() > m_n_accepted_points)
        {

            // Take channels that are considered as inliers
            // Solve using assumed inlier channels
            arma::mat result = solve_with_channel_subset(inlier_indices);
            auto residual = m_solver->get_signal_residual();
            LOG(DEBUG) << "Analysis result with all inliers: " << result;

            // Evaluate objective value which is error value calculated for assumed inliers
            auto objective_value = arma::as_scalar(objective(residual));
            LOG(DEBUG) << "Objective value: " << objective_value;

            // Update solution if objective value is best so far
            if (objective_value < lowest_objective_value)
            {
                LOG(DEBUG) << "Solution update";
                LOG(DEBUG) << "Round " << round;
                lowest_objective_value = objective_value;
                LOG(DEBUG) << "Lowest objective value so far: " << lowest_objective_value;
                m_x = result;
                LOG(DEBUG) << "Updated solution: " << m_x;

                // Stop iteration if target objective value is reached
                if (lowest_objective_value < m_objective_value_threshold)
                {
                    LOG(DEBUG) << "Object value threshold was reached at round " << round;
                    LOG(DEBUG) << "Iteration will be terminated";
                    return m_x;
                }
            }
        }
    }

    return m_x;
}