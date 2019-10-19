#ifndef NNGN_SOLVER_H
#define NNGN_SOLVER_H

#include "../gauss-newton/gn_solver.h"

class NNGNSolver : public GNSolver
{

public:
    using GNSolver::GNSolver;

    //! Solves f(x,L) = s using Gauss-newton with constaint x > 0
    arma::mat solve(const arma::mat &s) override;
};

#endif