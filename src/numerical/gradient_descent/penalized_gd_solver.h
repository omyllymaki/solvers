#ifndef PENALIZED_GD_SOLVER_H
#define PENALIZED_GD_SOLVER_H

#include "gd_solver.h"

class PenalizedGDSolver : public GDSolver
{

private:
    double m_penalty;

public:
    PenalizedGDSolver(){};

    PenalizedGDSolver(const arma::mat &L,
                   const double lr = 100.0,
                   const int max_iter = 10000,
                   const double termination_threshold = 0.000001,
                   const double penalty = 10000);

protected:
    arma::mat objective(arma::mat estimate, arma::mat expected) override;
};

#endif