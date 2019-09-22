#include <iostream>
#include <armadillo>
#include "src/ls_solver.h"

using arma::mat;
using arma::randn;
using arma::as_scalar;
using namespace std;


int main(int argc, char const *argv[])
{

    float lr = 1000;

    mat L, s, s_estimate, x, residual;
    L.load("/home/ossi/Repos/Personal/math/debug/samples/data/signals.txt");
    s.load("/home/ossi/Repos/Personal/math/debug/samples/data/sum_signal.txt");

    LSSolver solver = LSSolver(L);
    mat result = solver.solve(s);
    cout << result << endl;

    return 0;
}
