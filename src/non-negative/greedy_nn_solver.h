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
    GreedyNNSolver(){};

    GreedyNNSolver(std::shared_ptr<Solver> solver);

    GreedyNNSolver(arma::mat L);

    arma::mat solve(const arma::mat &s) override;

private:
    std::shared_ptr<Solver> m_solver;
};

#endif