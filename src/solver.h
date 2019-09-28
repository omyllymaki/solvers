#ifndef SOLVER_H
#define SOLVER_H

#include <iostream>
#include <armadillo>

//! Solver interface that needs to be inherited by individual solvers
//! Solvers solves x from equations of the form f(x, L) = s
//! s is observed signal and L is fixed set of coefficients
class Solver
{
public:

    //! Solves x, given signal
    //! @param signal - observed signal
    //! @returns solved x values
    virtual arma::mat solve(const arma::mat &signal) = 0;

    //! Like solve but solves multiple signals at one time
    virtual std::vector<arma::mat> solve_multiple(const std::vector <arma::mat> &signals);

    //! Estimated signal, calculated using solution x
    virtual arma::mat get_signal_estimate();

    //! Difference between estimated and observed signal
    virtual arma::mat get_signal_residual();

protected:
    arma::mat m_L;
    arma::mat m_s;
    arma::mat m_x;

    //! Signal model f
    //! Function that calculates signal, given x and L
    //! Abstract method that needs to be implemented by inheritors
    virtual arma::mat model(arma::mat x, arma::mat L) = 0;
};

#endif