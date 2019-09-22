#include "ls_solver.h"
#include "common.h"

using arma::mat;

LSSolver::LSSolver(const arma::mat &L)
{
    m_L_inv = calculate_pseudoinverse(L);
}

arma::mat LSSolver::solve(const arma::mat &s)
{
    return s * m_L_inv;
}