#ifndef NNLS_SOLVER_H
#define NNLS_SOLVER_H

#include "solver.h"

class NNLSSolver : public Solver
{

public:
    NNLSSolver(const arma::mat &L);

    arma::mat solve(const arma::mat &s) override;

    arma::mat get_signal_estimate() override;
};

#endif