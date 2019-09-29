#include <iostream>
#include <armadillo>

using namespace std;

arma::mat model(arma::mat x, arma::mat L)
{
    return x * L;
}

arma::mat objective(arma::mat estimate, arma::mat expected)
{
    arma::mat residual = estimate - expected;
    return sqrt(sum(pow(residual, 2), 1) / residual.n_elem);
}

arma::mat solve(const arma::mat &s, const arma::mat &L)
{

    int n_components = L.n_rows;
    int stdev_scaling_factor = 1;
    arma::mat best_guess = arma::zeros(1, n_components);
    // arma::mat stdevs = arma::randn(1, n_components);
    int n_candidates = 1000;
    int max_iter = 500;
    arma::mat s_estimate, obj_value, candidate, best_obj_value, stdevs;

    s_estimate = model(best_guess, L);
    best_obj_value = objective(s_estimate, s);

    for (size_t j = 0; j < max_iter; j++)
    {
        double factor = cos(j * M_PI / (2*max_iter));
        stdevs = stdev_scaling_factor * arma::randn(1, n_components) * factor;
        arma::mat obj_values, candidates;
        candidates.insert_rows(0, best_guess);
        obj_values.insert_rows(0, best_obj_value);

        for (size_t i = 1; i < n_candidates; i++)
        {
            candidate = best_guess + stdevs * arma::randn();
            s_estimate = model(candidate, L);
            obj_value = objective(s_estimate, s);
            obj_values.insert_rows(i, obj_value);
            candidates.insert_rows(i, candidate);
            // cout << obj_value << " " << candidate << endl;
        }
        //cout << obj_values.t() << endl;
        //cout << candidates << endl;
        int i_min = obj_values.index_min();
        best_obj_value = obj_values.row(i_min);
        best_guess = candidates.row(i_min);
        cout << i_min << " " << best_obj_value << endl;
        //cout << best_guess << endl;

    }

    return best_guess;
}

int main(int argc, char const *argv[])
{

    arma::mat L, s;
    L.load("./samples/data/signals.txt");
    s.load("./samples/data/sum_signal.txt");

    arma::mat result = solve(s, L);
    cout << result << endl;

    return 0;
}
