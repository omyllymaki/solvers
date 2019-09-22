#ifndef SOLVER_H
#define SOLVER_H

#include <armadillo>

class Solver
{
public:
    virtual arma::mat solve(const arma::mat &signal) = 0;
};

#endif