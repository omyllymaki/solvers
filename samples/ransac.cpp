#include "plotting.cpp"
#include "data_generation.cpp"
#include "../src/analytical/linear/ls_solver.h"
#include "../src/numerical/gauss-newton/gn_solver.h"
#include "../src/numerical/gradient_descent/gd_solver.h"
#include "../src/logging/easylogging++.h"
#include <math.h>
#include <armadillo>
#include <algorithm>
#include <random>
#include "../src/common.h"

using std::cout;
using std::endl;

INITIALIZE_EASYLOGGINGPP

template <typename T>
class RansacSolver
{
private:
    T m_solver;
    int m_n_channels;
    int m_n_max_iter;
    float m_accepted_error;
    int m_n_accepted_points;
    float m_objective_value_threshold;
    arma::mat m_s;
    arma::mat m_L;

    arma::mat solve_with_channel_subset(arma::uvec indices);

    arma::mat objective(arma::mat residual);

public:
    RansacSolver(T solver,
                 int n_channels,
                 float accepted_error,
                 int n_accepted_points,
                 float objective_value_threshold = 0.000001,
                 int max_iter = 1000);

    ~RansacSolver();

    arma::mat solve(const arma::mat &signal);
};

template <typename T>
RansacSolver<T>::RansacSolver(T solver,
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
}

template <typename T>
RansacSolver<T>::~RansacSolver()
{
}

template <typename T>
arma::mat RansacSolver<T>::solve_with_channel_subset(arma::uvec indices)
{
    arma::mat s_subset = m_s.elem(indices).t();
    arma::mat L_subset = m_L.cols(indices);
    m_solver.set_library(L_subset);
    arma::mat result = m_solver.solve(s_subset);
    return result;
}

template <typename T>
arma::mat RansacSolver<T>::objective(arma::mat residual) {
    return sqrt(sum(pow(residual, 2), 1) / residual.n_elem);
}

template <typename T>
arma::mat RansacSolver<T>::solve(const arma::mat &s)
{
    m_s = s;
    m_L = m_solver.get_library();
    arma::mat solution;
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
        m_solver.set_library(m_L);
        m_solver.set_signal(s);
        auto residual = m_solver.get_signal_residual();
        arma::uvec inlier_indices = arma::find(arma::abs(residual) < m_accepted_error);
        LOG(DEBUG) << "Number on inliers " << inlier_indices.size();

        // If number of assumed inliers is large enough, continue to evaluation
        if (inlier_indices.size() > m_n_accepted_points)
        {

            // Take channels that are considered as inliers
            // Solve using assumed inlier channels
            arma::mat result = solve_with_channel_subset(inlier_indices);
            auto residual = m_solver.get_signal_residual();
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
                solution = result;
                LOG(DEBUG) << "Updated solution: " << solution;

                // Stop iteration if target objective value is reached
                if (lowest_objective_value < m_objective_value_threshold)
                {
                    LOG(DEBUG) << "Object value threshold was reached at round " << round;
                    LOG(DEBUG) << "Iteration will be terminated";
                    return solution;
                }
            }
        }
    }

    return solution;
}

arma::mat WEIGHTS = {100, -20, 50, -0.5};

int main(int argc, char *argv[])
{
    el::Configurations conf("./logging-config.conf");
    el::Loggers::reconfigureLogger("default", conf);

    auto data_generator = DataGenerator();
    auto L = data_generator.generate_library();
    //auto s = data_generator.generate_noisy_linear_signal(WEIGHTS);
    auto s = data_generator.generate_linear_signal(WEIGHTS);

    std::vector<int> indices = {0, 5, 11, 13, 15, 18, 20, 25, 40, 60};
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0, 5);
    for (auto &&i : indices)
    {
        s[i] += distribution(generator);
    }

    LOG(INFO) << "True: " << WEIGHTS;

    // auto solver = GNSolver(L);
    // auto solver = LSSolver(L);
    auto solver = GDSolver(L);
    arma::mat result = solver.solve(s);
    LOG(INFO) << "Regular fit: " << result;

    int n_channels = 4;
    int n_max_iter = 1000;
    float accepted_error = 0.1;
    int n_accepted_points = 70;
    float objective_value_threshold = 0.0001;

    auto ransac_solver = RansacSolver<GDSolver>(solver, n_channels, accepted_error, n_accepted_points, objective_value_threshold, n_max_iter);
    auto solution = ransac_solver.solve(s);
    LOG(INFO) << "RANSAC fit: " << solution;

    return 0;
}