#ifndef LINEAR_NNLS_SOLVER_H
#define LINEAR_NNLS_SOLVER_H

#include "../linear/ls_solver.h"

class LinearNNLSSolver : public LSSolver
{

public:
    using LSSolver::LSSolver;

    arma::mat solve(const arma::mat &s) override;
};

#endif