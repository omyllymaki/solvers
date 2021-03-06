#include "ls_solver.h"
#include "../../common.h"

using arma::mat;

LSSolver::LSSolver(const arma::mat &L)
{
    m_L = L;
    m_L_inv = calculate_svd_inverse(L);
}

arma::mat LSSolver::solve(const arma::mat &s)
{
    m_s = s;
    m_x = s * m_L_inv;
    return m_x;
}

void LSSolver::set_library(arma::mat L)
{
    m_L = L;
    m_L_inv = calculate_svd_inverse(L);
}