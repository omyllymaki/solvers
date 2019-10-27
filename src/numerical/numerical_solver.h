#ifndef NUMERICAL_SOLVER_H
#define NUMERICAL_SOLVER_H

#include "../solver.h"

class NumericalSolver : public Solver
{

protected:
    int m_max_iter;
    size_t m_round;

    virtual void update_solution() = 0;

    virtual bool is_termination_condition_filled() = 0;

    virtual arma::mat objective(arma::mat estimate, arma::mat expected) = 0;

public:
    NumericalSolver(){};

    //! Set signal model f in f(x, L) = s
    //! This method can be used to replace default model of solver
    virtual void set_model(model_wrapper f_model);
};

#endif