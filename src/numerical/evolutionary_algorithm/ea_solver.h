#ifndef EA_SOLVER_H
#define EA_SOLVER_H

#include "../numerical_solver.h"

//! Solver based on evolutionary algorithm (EA) that is generic population-based metaheuristic optimization algorithm.
//! The algorithm works by generating population, selecting best candidates, and the genarating new popolation based on best candidates.
//! At fist, population is random.
//! Candidate selection is based on objective value which is calculated for every candidate.
//! The algorithm doesn't make any assumptions about shape of the cost function, in contrast to gradient based methods.
//! Thus it can be used to solve various kind of problems.
//! The drawback is that the algorithm provides only approximate solutions and it is computionally expensive. 
class EASolver : public NumericalSolver
{

protected:
    int m_n_candidates, m_n_no_change_threshold, m_min_index, m_no_change_counter;
    arma::mat m_best_obj_value, m_stdev_scaling_factors, m_stdevs, m_obj_values, m_candidates, m_init_guess;
    double m_objective_threshold;

    virtual arma::mat objective(arma::mat estimate, arma::mat expected) override;

    virtual bool is_termination_condition_filled() override;

    virtual void update_solution() override;

    virtual void update_stdevs();

    virtual void generate_and_test_candidates();

    virtual void initialize();

public:
    EASolver(){};

    //! Solver initialization.
    //! @param L - Library used to fit model f(x,L) = s.
    //! @param n_candidates - Number of candidates generated in every iteration.
    //! @param max_iter - Maximum number of iterations.
    //! @param objective_threshold - Threshold value for objective. The iteration is terminated when threshold value is reached.
    //! @param n_no_change_threshold - Threshold value for number of iterations that do not change objective. The iteration is terminated when threshold value is reached.
    //! @param stdev_scaling_factor - Scaling factor used to scale stdevs in every iteration. Stdevs are used to generate new population around best candidate from previous population.
    EASolver(const arma::mat &L,
             const int n_candidates = 500,
             const int max_iter = 1000,
             const double objective_threshold = 0.00001,
             const int n_no_change_threshold = 50,
             const double stdev_scaling_factor = 1);

    //! Solver initialization.
    //! @param L - Library used to fit model f(x,L) = s.
    //! @param stdev_scaling_factors - Scaling factor for every individual element in solution x. Scaling factor used to scale stdevs in every iteration. Stdevs are used to generate new population around best candidate from previous population.
    //! @param n_candidates - Number of candidates generated in every iteration.
    //! @param max_iter - Maximum number of iterations.
    //! @param objective_threshold - Threshold value for objective. The iteration is terminated when threshold value is reached.
    //! @param n_no_change_threshold - Threshold value for number of iterations that do not change objective. The iteration is terminated when threshold value is reached.
    EASolver(const arma::mat &L,
             const arma::mat stdev_scaling_factors,
             const int n_candidates = 500,
             const int max_iter = 1000,
             const double objective_threshold = 0.00001,
             const int n_no_change_threshold = 50);

    arma::mat solve(const arma::mat &s) override;
};

#endif