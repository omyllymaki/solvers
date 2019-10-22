#ifndef GREEDY_NN_SOLVER_H
#define GREEDY_NN_SOLVER_H

#include "../solver.h"
#include "../common.h"
#include <iostream>
#include <vector>

using arma::mat;
using arma::zeros;
using std::vector;

template <typename T>
class GreedyNNSolver : public Solver
{

public:
    T m_solver;

    GreedyNNSolver() {};

    GreedyNNSolver(T solver);

    arma::mat solve(const arma::mat &s) override; 
};

#endif 