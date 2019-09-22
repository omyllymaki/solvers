
#include "utils.cpp"
#include "../src/ls_solver.h"
#include "../src/nnls_solver.h"
#include "../src/gd_linear_solver.h"
#include <armadillo>

using namespace arma;

int main()
{
    mat L, s;
    L.load("./data/signals.txt");
    s.load("./data/sum_signal.txt");

    LSSolver ls_solver = LSSolver(L);
    mat result1 = ls_solver.solve(s);
    print("LS fit", false);
    print(result1);

    NNLSSolver nnls_solver = NNLSSolver(L);
    mat result2 = nnls_solver.solve(s);
    print("NNLS fit", false);
    print(result2);

    GDLinearSolver gd_linear_solver = GDLinearSolver(L);
    mat result3 = gd_linear_solver.solve(s);
    print("GD linear fit", false);
    print(result3);

    return 0;
}