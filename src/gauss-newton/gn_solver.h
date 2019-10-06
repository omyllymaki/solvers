#ifndef GN_SOLVER_H
#define GN_SOLVER_H

#include "../solver.h"

class GNSolver : public Solver
{

protected:
    double m_termination_threshold;
    int m_max_iter;
    size_t m_round;
    const double m_x_delta = 0.000001;
    arma::mat m_objective_prev, m_objective, m_gradient, m_residual;

    virtual arma::mat model(arma::mat x, arma::mat L) override;

    virtual void update_solution();

    virtual void update_gradient();

    virtual bool is_termination_condition_filled();

    virtual void update_objective();

    virtual void update_residual();

    virtual arma::mat objective(arma::mat estimate, arma::mat expected);

public:
    GNSolver(const arma::mat &L,
             const int max_iter = 100,
             const double termination_threshold = 0.000001);

    arma::mat solve(const arma::mat &s) override;
};

#endif