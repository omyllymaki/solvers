#ifndef NNLS
#define NNLS

#include <armadillo>

using namespace arma;

mat nnls_fit(mat L, mat s);

#endif