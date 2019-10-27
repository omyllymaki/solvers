#ifndef GD_SOLVER_H
#define GD_SOLVER_H

#include "../numerical_solver.h"

class GDSolver : public NumericalSolver
{

protected:
    double m_lr, m_termination_threshold;
    static constexpr double m_x_delta = 0.000001;
    arma::mat m_objective_prev, m_objective, m_gradient;

    virtual void update_solution() override;

    virtual bool is_termination_condition_filled() override;

    virtual arma::mat objective(arma::mat estimate, arma::mat expected) override;

    virtual void update_learning_rate();

    virtual void update_gradient();

public:
    GDSolver(){};

    GDSolver(const arma::mat &L,
             const double lr = 100.0,
             const int max_iter = 10000,
             const double termination_threshold = 0.000001);

    arma::mat solve(const arma::mat &s) override;

    arma::mat get_objective_value();

    virtual void set_learning_rate(double lr);

    //! Find optimal learning rate from provided lr_array
    //! Optimal learning rate is found by testing every candidate in lr_array
    //! Optimal learning rate is the one which has lowest objective value after n_iter iterations
    double find_optimal_lr(const arma::mat &s, arma::mat lr_array, int n_iter = 10);

    //! Find optimal learning rate from lr_array
    //! lr_array is either linearly spaced [lb, ub] or logarithmically spaced [10^lb, 10^ub] array with n_candidates points
    //! Optimal learning rate is found by testing every candidate in lr_array
    //! Optimal learning rate is the one which has lowest objective value after n_iter iterations
    double find_optimal_lr(const arma::mat &s, double lb = -5, double ub = 5, int n_candidates = 20, int n_iter = 10, std::string method = "log");
};

#endif