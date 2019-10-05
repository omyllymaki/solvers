#ifndef ROBUST_EA_SOLVER_H
#define ROBUST_EA_SOLVER_H

#include "ea_solver.h"

class RobustEASolver : public EASolver
{

protected:
    arma::mat objective(arma::mat estimate, arma::mat expected) override;

public:
    using EASolver::EASolver;
};

#endif