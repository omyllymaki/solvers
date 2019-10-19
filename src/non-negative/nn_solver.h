#ifndef NN_SOLVER_H
#define NN_SOLVER_H

#include "../solver.h"

class NNSolver : public Solver
{
public:
    arma::mat nn_solve(const arma::mat &s);
};

#endif