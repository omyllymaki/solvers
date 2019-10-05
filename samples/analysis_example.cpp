#include "utils.cpp"
#include "../src/linear/ls_solver.h"
#include "../src/linear/nnls_solver.h"
#include "../src/gradient_descent/gd_linear_solver.h"
#include "../src/evolutionary_algorithm/ea_solver.h"
#include <math.h>
#include <armadillo>

using namespace arma;

mat WEIGHTS = {100, -20, 5.0, -0.5};
mat CENTERS = {20.0, 35.0, 40.0, 45.0};
mat SIGMAS = {3.0, 10.0, 5.0, 2.0};
mat CHANNELS = linspace(0, 99, 100);

mat generate_signal_matrix(mat channels, mat centers, mat sigmas)
{
    int n_components = centers.n_elem;
    mat signals = zeros(0, channels.n_elem);
    for (int i = 0; i < n_components; ++i)
    {
        float center = centers[i];
        float sigma = sigmas[i];
        mat signal = calculate_gaussian(sigma, center, channels);
        signals = join_vert(signals, signal.t());
    };
    return signals;
}

int main()
{
    mat L = generate_signal_matrix(CHANNELS, CENTERS, SIGMAS);
    mat s = WEIGHTS * L;

    LSSolver ls_solver = LSSolver(L);
    mat result1 = ls_solver.solve(s);
    print("LS fit", false);
    print(result1);

    NNLSSolver nnls_solver = NNLSSolver(L);
    mat result2 = nnls_solver.solve(s);
    print("NNLS fit", false);
    print(result2);

    GDLinearSolver gd_linear_solver = GDLinearSolver(L, 5000, 500);
    mat result3 = gd_linear_solver.solve(s);
    print("GD linear fit", false);
    print(result3);

    EASolver ea_solver = EASolver(L);
    mat result4 = ea_solver.solve(s);
    print("EA fit", false);
    print(result4);

#ifdef PLOT_FIGURES
    plot_arma_mat(L, 1, "Pure components");

    plot_arma_vec(s, 2, "EA solver solution");
    plot_arma_vec(ea_solver.get_signal_estimate(), 2);
    plot_arma_vec(ea_solver.get_signal_residual(), 2);
    plt::show();
#endif

    return 0;
}