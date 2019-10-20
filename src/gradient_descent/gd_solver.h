#ifndef GD_SOLVER_H
#define GD_SOLVER_H

#include "../solver.h"

class GDSolver : public Solver
{

protected:
    double m_lr, m_termination_threshold;
    int m_max_iter;
    size_t m_round;
    static constexpr double m_x_delta = 0.000001;
    arma::mat m_objective_prev, m_objective, m_gradient;

    virtual void update_gradient();

    virtual void update_solution();

    virtual void update_learning_rate();

    virtual bool is_termination_condition_filled();

    virtual arma::mat objective(arma::mat estimate, arma::mat expected);

public:
    GDSolver(){};

    GDSolver(const arma::mat &L,
             const double lr = 100.0,
             const int max_iter = 10000,
             const double termination_threshold = 0.000001);

    arma::mat solve(const arma::mat &s) override;

    arma::mat get_objective_value();

    virtual void set_learning_rate(double lr);

    double find_optimal_lr(const arma::mat &s, arma::mat lr_array, int n_iter = 10);

    double find_optimal_lr(const arma::mat &s, double lb = -5, double ub = 5, int n_candidates = 20, int n_iter = 10, std::string method = "log");
};

#endif