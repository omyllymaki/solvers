#include <armadillo>

using namespace arma;

mat create_signals()
{
    mat signals = {{1, 0, 0, 0},
                   {0, 1, 0, 1},
                   {1, 1, 0, 0}};

    return signals;
}

mat sum_signal(mat weights, mat signals)
{
    return weights * signals;
}

bool is_equal(mat x, mat y, float threshold = 0.001)
{
    return approx_equal(x, y, "absdiff", threshold);
}