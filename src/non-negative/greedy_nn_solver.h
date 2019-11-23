#ifndef GREEDY_NN_SOLVER_H
#define GREEDY_NN_SOLVER_H

#include "../solver.h"
#include "../common.h"
#include <iostream>
#include <vector>
#include <memory>

using arma::mat;
using arma::zeros;
using std::vector;

//! Greedy non-negative solver.
//! This method constaints fit so that every element in solution is larger or equal to zero: xi > 0 for every xi in x.
//! The method is greedy in a sense that in every iteration step the algorithm selects most negative element in solution and replaces that with zero.
//! Due to greediness, the solution might not be globally optimal.
class GreedyNNSolver : public Solver
{

public:
    GreedyNNSolver(){};

    //! Solver initialization.
    //! @param solver - Any other solver that uses solver interface. The solver is used to fit model f(x,L) = s.
    GreedyNNSolver(std::shared_ptr<Solver> solver);

    //! Solver initialization with default solver.
    //! Solver uses linear LS model for fitting.
    //! @param L - Library.
    GreedyNNSolver(arma::mat L);

    arma::mat solve(const arma::mat &s) override;

private:
    std::shared_ptr<Solver> m_solver;
};

#endif