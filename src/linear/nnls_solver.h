#ifndef NNLS_SOLVER_H
#define NNLS_SOLVER_H

#include "../solver.h"

class NNLSSolver : public Solver
{

public:
    NNLSSolver(const arma::mat &L);

    arma::mat solve(const arma::mat &s) override;

protected:

    arma::mat model(arma::mat x, arma::mat L) override;
};

#endif