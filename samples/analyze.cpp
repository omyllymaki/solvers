
#include "utils.cpp"
#include "../src/nnls.cpp"
#include <armadillo>

using namespace arma;

int main()
{
    mat signals, sum_signal;
    signals.load("./data/signals.txt");
    sum_signal.load("./data/sum_signal.txt");

    mat result_ls = ls_fit(signals, sum_signal);
    print("LS fit: ", false);
    print(result_ls);

    mat result_nnls = nnls_fit(signals, sum_signal);
    print("NNLS fit: ", false);
    print(result_nnls);

    return 0;
}