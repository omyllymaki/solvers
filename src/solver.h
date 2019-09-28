#ifndef SOLVER_H
#define SOLVER_H

#include <armadillo>

class Solver
{
public:
    virtual arma::mat solve(const arma::mat &signal) = 0;

    virtual arma::mat get_signal_estimate();

    virtual arma::mat get_signal_residual();

protected:
    arma::mat m_L;
    arma::mat m_s;
    arma::mat m_x;

    virtual arma::mat model(arma::mat x, arma::mat L) = 0;
};

#endif