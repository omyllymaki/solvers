#include <iostream>
#include "ls.h"

using namespace arma;
using std::vector;

mat ls_fit(mat L, mat s)
{
    /* 
    Finds best linear least squares fit to system of linear equations: x*L = s.
    
    Returns fitted x values as solution.
    */
    mat L_inv = calculate_pseudoinverse(L);
    return s * L_inv;
}