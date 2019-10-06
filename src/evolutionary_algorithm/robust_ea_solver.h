#ifndef ROBUST_EA_SOLVER_H
#define ROBUST_EA_SOLVER_H

#include "ea_solver.h"

class RobustEASolver : public EASolver
{

protected:
    float m_rejection_threshold = 0.05;

    arma::mat objective(arma::mat estimate, arma::mat expected) override;

public:
    using EASolver::EASolver;

    void set_rejection_threshold(double value);
};

#endif