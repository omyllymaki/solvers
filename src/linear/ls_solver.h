#ifndef LS_SOLVER_H
#define LS_SOLVER_H

#include "../solver.h"

class LSSolver : public Solver
{

private:
    arma::mat m_L_inv;

public:
    LSSolver(const arma::mat &L);

    arma::mat solve(const arma::mat &s) override;

protected:

    arma::mat model(arma::mat x, arma::mat L) override;
};

#endif