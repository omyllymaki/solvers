#ifndef NNLS_SOLVER_H
#define NNLS_SOLVER_H

#include "solver.h"

class NNLSSolver : Solver
{

private:
    arma::mat m_L;

public:
    NNLSSolver(const arma::mat &L);

    arma::mat solve(const arma::mat &s) override;
};

#endif