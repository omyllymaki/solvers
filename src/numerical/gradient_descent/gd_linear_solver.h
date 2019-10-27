#ifndef GD_LINEAR_SOLVER_H
#define GD_LINEAR_SOLVER_H

#include "gd_solver.h"

class GDLinearSolver : public GDSolver
{

private:
    double m_lr_max;

public:

    GDLinearSolver() {};

    GDLinearSolver(const arma::mat &L,
                   const double lr = 0.1,
                   const int max_iter = 5000);

protected:

    arma::mat objective(arma::mat estimate, arma::mat expected) override;

    bool is_termination_condition_filled() override;

    void update_learning_rate() override;
};

#endif