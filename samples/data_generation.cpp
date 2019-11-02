#include <math.h>
#include <armadillo>

arma::mat CENTERS = {20.0, 35.0, 40.0, 45.0};
arma::mat SIGMAS = {3.0, 10.0, 5.0, 2.0};
arma::mat CHANNELS = arma::linspace(0, 99, 100);

const double EulerConstant = std::exp(1.0);

class DataGenerator
{

public:
    arma::mat m_L;

    DataGenerator()
    {
        m_L = generate_library();
    }

    arma::mat generate_library()
    {
        return generate_signal_matrix(CHANNELS, CENTERS, SIGMAS);
    };

    arma::mat generate_signal(arma::mat weights, double noise = 0.00)
    {
        arma::mat signal = get_linear_signal(weights);
        return signal + noise * arma::randn(1, signal.n_elem);
    }

    arma::mat generate_signal(arma::mat weights, std::vector<int> indices, double sigma_square = 10, double noise = 0.00)
    {
        arma::mat signal = get_linear_signal(weights);
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0, sigma_square);
        for (auto &&i : indices)
        {
            signal[i] += distribution(generator);
        }
        return signal + noise * arma::randn(1, signal.n_elem);
    }

    template <typename Function>
    arma::mat generate_signal(arma::mat weights, arma::mat library, Function model, float noise = 0)
    {
        arma::mat signal = model(weights, library);
        return signal + noise * arma::randn(1, signal.n_elem);
    }

private:
    arma::mat generate_signal_matrix(arma::mat channels, arma::mat centers, arma::mat sigmas)
    {
        int n_components = centers.n_elem;
        arma::mat signals = arma::zeros(0, channels.n_elem);
        for (int i = 0; i < n_components; ++i)
        {
            float center = centers[i];
            float sigma = sigmas[i];
            arma::mat signal = calculate_gaussian(sigma, center, channels);
            signals = join_vert(signals, signal.t());
        };
        return signals;
    }

    arma::mat get_linear_signal(arma::mat x)
    {
        return x * m_L;
    }

    arma::mat calculate_gaussian(float sigma, float center, arma::mat x)
    {
        float scaling_factor = 1 / (sigma * sqrt(2 * M_PI));
        arma::mat exponent = -pow((x - center) / sigma, 2) / 2;
        arma::mat result = scaling_factor * pow_to_vector(EulerConstant, exponent);
        return result;
    }

    arma::mat pow_to_vector(float x, arma::mat y)
    {
        arma::mat result;
        result.copy_size(y);
        for (std::size_t i = 0; i < result.n_elem; ++i)
        {
            result[i] = std::pow(x, y[i]);
        }
        return result;
    }
};
