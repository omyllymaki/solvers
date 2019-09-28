#include "utils.cpp"
#include <math.h>
#include <armadillo>

using namespace arma;

mat WEIGHTS = {1.0, -2.0, 5.0, -0.5};
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
    mat signals = generate_signal_matrix(CHANNELS, CENTERS, SIGMAS);
    mat sum_signal = WEIGHTS * signals;

    create_dir_if_not_exist("data");
    signals.save("./data/signals.txt", arma_ascii);
    sum_signal.save("./data/sum_signal.txt", arma_ascii);

    #ifdef PLOT_FIGURES
        plot_arma_vec(sum_signal, 1);
        plot_arma_mat(signals, 2);
        plt::show();
    #endif

    return 0;
}