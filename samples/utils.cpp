#include <iostream>
#include <bits/stdc++.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <string>
#include <math.h>
#include <armadillo>

using namespace arma;
using namespace std;

template <typename T>
void print(T x, bool add_endl = true)
{
    if (add_endl)
    {
        cout << x << endl;
    }
    else
    {
        cout << x;
    }
}

void create_dir_if_not_exist(const char *dir_path)
{
    mkdir(dir_path, 0777);
}

const double EulerConstant = std::exp(1.0);

mat pow_to_vector(float x, mat y)
{
    mat result;
    result.copy_size(y);
    for (std::size_t i = 0; i < result.n_elem; ++i)
    {
        result[i] = std::pow(x, y[i]);
    }
    return result;
}

mat calculate_gaussian(float sigma, float center, mat x)
{
    float scaling_factor = 1 / (sigma * sqrt(2 * M_PI));
    mat exponent = -pow((x - center) / sigma, 2) / 2;
    mat result = scaling_factor * pow_to_vector(EulerConstant, exponent);
    return result;
}