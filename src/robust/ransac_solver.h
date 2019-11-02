#include <armadillo>

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