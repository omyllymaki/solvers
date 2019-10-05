#ifndef EA_SOLVER_H
#define EA_SOLVER_H

#include "../solver.h"

class EASolver : public Solver
{

protected:
    int m_n_candidates, m_max_iter, m_n_no_change_threshold, m_min_index, m_no_change_counter;
    arma::mat m_best_obj_value, m_stdev_scaling_factors, m_stdevs, m_obj_values, m_candidates;
    size_t m_round;
    double m_objective_threshold;

    virtual arma::mat objective(arma::mat estimate, arma::mat expected);

    virtual arma::mat model(arma::mat x, arma::mat L) override;

    virtual bool is_termination_condition_filled();

    virtual void update_stdevs();

    virtual void generate_and_test_candidates();

    virtual void update_solution();

public:
    EASolver(const arma::mat &L,
             const int n_candidates = 500,
             const int max_iter = 1000,
             const double objective_threshold = 0.00001,
             const int n_no_change_threshold = 50,
             const double stdev_scaling_factor = 1);

    EASolver(const arma::mat &L,
             const arma::mat stdev_scaling_factors,
             const int n_candidates = 500,
             const int max_iter = 1000,
             const double objective_threshold = 0.00001,
             const int n_no_change_threshold = 50);

    arma::mat solve(const arma::mat &s) override;
};

#endif