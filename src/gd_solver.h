#ifndef GD_SOLVER_H
#define GD_SOLVER_H

#include "solver.h"

class GDSolver : public Solver
{

private:
    arma::mat m_L;
    double m_lr;
    int m_max_iter;
    double m_termination_threshold;

public:
    GDSolver(const arma::mat &L,
             const double lr = 100.0,
             const int max_iter = 10000,
             const double termination_threshold = 0.000001);

    arma::mat solve(const arma::mat &s);

    virtual arma::mat f_objective(arma::mat estimate, arma::mat expected) = 0;

    virtual arma::mat f_model(arma::mat x, arma::mat L) = 0;
};

#endif