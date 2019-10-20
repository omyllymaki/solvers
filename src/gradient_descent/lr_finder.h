#ifndef LR_FINDER_H
#define LR_FINDER_H

#include <armadillo>

template <typename T>
class LrFinder
{

public:
    T m_solver;

    LrFinder(T solver);

    double find_best_lr(arma::mat lrArray, const arma::mat &signal);
};

#endif