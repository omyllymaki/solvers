#include <iostream>
#include <armadillo>
#include "src/ls_solver.h"
#include "src/nnls_solver.h"
#include "src/gd_solver.h"
#include "src/gd_linear_solver.h"

using arma::as_scalar;
using arma::mat;
using arma::randn;
using namespace std;

int main(int argc, char const *argv[])
{

    float lr = 1000;

    mat L, s, s_estimate, x, residual;
    L.load("/home/ossi/Repos/Personal/math/debug/samples/data/signals.txt");
    s.load("/home/ossi/Repos/Personal/math/debug/samples/data/sum_signal.txt");

    LSSolver ls_solver = LSSolver(L);
    mat result1 = ls_solver.solve(s);
    cout << result1 << endl;

    NNLSSolver nnls_solver = NNLSSolver(L);
    mat result2 = nnls_solver.solve(s);
    cout << result2 << endl;

    GDLinearSolver gd_linear_solver = GDLinearSolver(L);
    mat result3 = gd_linear_solver.solve(s);
    cout << result3 << endl;

    return 0;
}
