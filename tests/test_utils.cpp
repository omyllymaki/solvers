#include <armadillo>

using namespace arma;

arma::mat COMMON_TEST_WEIGHTS = {
    {1, 1, 2},
    {0, 0, 0},
    {1, 0, 10},
    {5000, 1, 10},
    {-10, -3, -7},
    {-10, 3, -7},
};

arma::mat SIGNALS = {
    {1, 0, 0, 0},
    {0, 1, 0, 1},
    {1, 1, 0, 0},
};

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

template <typename T>
class SolverTester
{
public:
    T m_solver;
    arma::mat m_signals;

    SolverTester(T solver)
    {
        m_solver = solver;
        set_up();
    }

    void set_up()
    {
        m_signals = SIGNALS;
        m_solver.set_library(m_signals);
    }

    mat model(mat weights, mat signals)
    {
        return weights * signals;
    }

    void test_solve(arma::mat weights, float tolerance = 0.001)
    {
        mat signal = model(weights, m_signals);
        mat result = m_solver.solve(signal);
        BOOST_TEST_MESSAGE(boost::unit_test::framework::current_test_case().p_name);
        BOOST_TEST_MESSAGE("Expected: " << weights);
        BOOST_TEST_MESSAGE("Actual: " << result);
        BOOST_CHECK(is_equal(weights, result, tolerance));
    }

    void test_multiple_solve(arma::mat weights, float tolerance = 0.001)
    {
        for (int row = 0; row < weights.n_rows; row++)
        {
            arma::mat test_weights = weights.row(row);
            set_up();
            test_solve(test_weights, tolerance);
        }
    }

    void test_state(arma::mat weights, float tolerance = 0.001)
    {
        // Test that solver produces same result without initialization of solver between
        // State of the solver should not affect to result
        set_up();
        test_solve(weights, tolerance);
        test_solve(weights, tolerance);
    }

    void test_common(float tolerance = 0.001)
    {
        arma::mat weights = {1, 1, 2};
        test_state(weights, tolerance);
        test_multiple_solve(COMMON_TEST_WEIGHTS, tolerance);
    }
};