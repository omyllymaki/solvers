#include <armadillo>
#include "../solver.h"

//! Random sample consensus (RANSAC) solver.
//! This is an iterative method to estimate parameters of a mathematical model from a set of observed data that contains outliers.
//! Responsibility of RANSAC solver is to make model robust against outliers.
//! This means that RANSAC solver itself doensn't make fit. Instead, it uses any other solver that uses solvers interface. 
template <typename T>
class RansacSolver : public Solver
{
private:
    T m_solver;
    int m_n_channels;
    int m_n_max_iter;
    float m_accepted_error;
    int m_n_accepted_points;
    float m_objective_value_threshold;

    arma::mat solve_with_channel_subset(arma::uvec indices);

    arma::mat objective(arma::mat residual);

public:
    //! Solver initialization
    //! @param solver - Any other solver that uses solver interface. The solver is used to fit model f(x,L) = s.
    //! @param n_channels - Number of channels used to construct initial model. Usually this is minimum number of channels needed to fit model.
    //! @param accepted_error - Error threshold. Channels with errors smaller than accepted error are considered as inliers.
    //! @param n_accepetd_points - Minimum number of inliers (channels) needed in order to consider model as valid
    //! @param objective_value_threshold - Threshold value for objective. Iteration is terminated when threshold value is reached.
    //! @param max_iter - Max number of iterations allowed.
    RansacSolver(T solver,
                 int n_channels,
                 float accepted_error,
                 int n_accepted_points,
                 float objective_value_threshold = 0.000001,
                 int max_iter = 1000);

    ~RansacSolver();

    arma::mat solve(const arma::mat &signal) override;
};