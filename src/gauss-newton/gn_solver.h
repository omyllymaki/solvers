#ifndef GN_SOLVER_H
#define GN_SOLVER_H

#include "../non-negative/nn_solver.h"

class GNSolver : public NNSolver
{

protected:
    double m_termination_threshold;
    int m_max_iter;
    size_t m_round;
    static constexpr double m_x_delta = 0.000001;
    arma::mat m_objective_prev, m_objective, m_jacobian, m_residual;

    virtual void update_solution();

    virtual void update_jacobian();

    virtual bool is_termination_condition_filled();

    virtual void update_objective();

    virtual void update_residual();

    virtual arma::mat objective(arma::mat estimate, arma::mat expected);

public:

    GNSolver() {};

    GNSolver(const arma::mat &L,
             const int max_iter = 100,
             const double termination_threshold = 0.000001);

    virtual arma::mat solve(const arma::mat &s) override;
};

#endif