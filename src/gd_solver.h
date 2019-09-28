#ifndef GD_SOLVER_H
#define GD_SOLVER_H

#include "solver.h"

class GDSolver : public Solver
{

protected:
    double m_lr;
    int m_max_iter;
    double m_termination_threshold;
    const double m_x_delta = 0.000001;
    arma::mat m_objective_prev, m_x, m_objective, m_s, m_gradient, m_L;

    virtual void update_gradient();

    virtual void update_solution();

    virtual bool is_termination_condition_filled();

    virtual arma::mat f_objective(arma::mat estimate, arma::mat expected) = 0;

    virtual arma::mat f_model(arma::mat x, arma::mat L) = 0;

public:
    GDSolver(const arma::mat &L,
             const double lr = 100.0,
             const int max_iter = 10000,
             const double termination_threshold = 0.000001);

    arma::mat solve(const arma::mat &s);
};

#endif