#ifndef PENALIZED_GD_SOLVER_H
#define PENALIZED_GD_SOLVER_H

#include "gd_solver.h"

//! Gradient descent solver with penalty.
//! Penalty is added to objective function in order to constrain fit.
//! In this class, penalty is given for negative values.
class PenalizedGDSolver : public GDSolver
{

private:
    double m_penalty;

public:
    PenalizedGDSolver(){};

    //! Solver initialization.
    //! @param L - Library used to fit model f(x,L) = s.
    //! @param lr - Learning rate. This is the main tuning parameter and affects to convergence and optimization time. 
    //! @param max_iter - Maximum number of iterations.
    //! @param termination_threshold - Threshold value for objective. The iteration is terminated when threshold value is reached.
    //! @param penalty - Amount of penalty given for negative values. 
    PenalizedGDSolver(const arma::mat &L,
                   const double lr = 100.0,
                   const int max_iter = 10000,
                   const double termination_threshold = 0.000001,
                   const double penalty = 10000);

protected:
    arma::mat objective(arma::mat estimate, arma::mat expected) override;
};

#endif