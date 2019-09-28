#ifndef LS_SOLVER_H
#define LS_SOLVER_H

#include "solver.h"

class LSSolver : Solver
{

private:
    arma::mat m_L_inv;

public:
    LSSolver(const arma::mat &L);

    arma::mat solve(const arma::mat &s) override;

    arma::mat get_signal_estimate() override;
};

#endif