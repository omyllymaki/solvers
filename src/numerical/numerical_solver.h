#ifndef NUMERICAL_SOLVER_H
#define NUMERICAL_SOLVER_H

#include "../solver.h"

//! Base NumericalSolver class
//! Provides interface and some common methods for all numerical solvers
class NumericalSolver : public Solver
{

protected:
    //! Maximum number of iterations
    int m_max_iter;

    //! Current iteration round
    size_t m_round;

    //! Method to update solution m_x in every iteration round
    //! Abstract method that needs to be implemented by inheritors
    virtual void update_solution() = 0;

    //! Method to check termination criteria
    //! Should return true if iteration needs to be terminated
    //! Abstract method that needs to be implemented by inheritors
    virtual bool is_termination_condition_filled() = 0;

    //! Objective function that needs to be minimized
    //! Takes estimate and expected values; returns value that needs to minimized by solver
    //! Abstract method that needs to be implemented by inheritors
    virtual arma::mat objective(arma::mat estimate, arma::mat expected) = 0;

public:
    NumericalSolver(){};

    //! Set signal model f in f(x, L) = s
    //! This method can be used to replace default model of solver
    virtual void set_model(model_wrapper f_model);
};

#endif