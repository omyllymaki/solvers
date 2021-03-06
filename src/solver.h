#ifndef SOLVER_H
#define SOLVER_H

#include <iostream>
#include <armadillo>
#include <functional>

using model_wrapper = std::function<arma::mat(arma::mat, arma::mat)>;

//! Base Solver class.
//! Provides interface and some common methods for all solvers.
//! Solvers solves x from equations of the form f(x, L) = s.
//! s is observed signal.
//! L is library which is fixed set of coefficients.
//! function f is signal model.
class Solver
{
public:
    //! Solves x, given signal.
    //! @param signal - observed signal
    //! @returns solved x values
    virtual arma::mat solve(const arma::mat &signal) = 0;

    //! Like solve but solves multiple signals at one time.
    virtual std::vector<arma::mat> solve_multiple(const std::vector<arma::mat> &signals);

    //! Estimated signal, calculated using solution x.
    virtual arma::mat get_signal_estimate();

    //! Difference between estimated and observed signal.
    virtual arma::mat get_signal_residual();

    //! Set library L in f(x, L) = s.
    virtual void set_library(arma::mat L);

    //! get library L in f(x, L) = s.
    arma::mat get_library();

    //! Set signal s in f(x, L) = s.
    virtual void set_signal(arma::mat s);

    //! Get model f in f(x, L) = s.
    model_wrapper get_model();

protected:
    arma::mat m_L;
    arma::mat m_s;
    arma::mat m_x;
    model_wrapper model = linear_model; // Default model

    //! Linear signal model, f = x*L.
    static arma::mat linear_model(arma::mat x, arma::mat L);
};

#endif