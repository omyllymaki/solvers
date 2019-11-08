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

class GreedyNNSolver : public Solver
{

public:
    std::shared_ptr<Solver> m_solver;

    GreedyNNSolver(){};

    GreedyNNSolver(std::shared_ptr<Solver> solver);

    arma::mat solve(const arma::mat &s) override;
};

#endif