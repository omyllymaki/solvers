#ifndef LS_SOLVER_H
#define LS_SOLVER_H

#include "../solver.h"

class LSSolver : public Solver
{

private:
    arma::mat m_L_inv;

public:

    LSSolver() {};
    
    LSSolver(const arma::mat &L);

    virtual arma::mat solve(const arma::mat &s) override;

    void set_library(arma::mat L) override;
};

#endif