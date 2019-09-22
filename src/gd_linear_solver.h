#ifndef GD_LINEAR_SOLVER_H
#define GD_LINEAR_SOLVER_H

#include "gd_solver.h"

class GDLinearSolver : public GDSolver
{

public:
    GDLinearSolver(const arma::mat &L,
                   const double lr = 100.0,
                   const int max_iter = 10000,
                   const double termination_threshold = 0.000001);

    arma::mat f_objective(arma::mat estimate, arma::mat expected) override;

    arma::mat f_model(arma::mat x, arma::mat L) override;
};

#endif