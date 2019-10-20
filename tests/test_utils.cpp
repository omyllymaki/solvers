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

template <typename T>
struct TestFixture
{
    mat m_signals;
    T m_solver;

    TestFixture()
    {
        m_signals = create_signals();
        m_solver.set_library(m_signals);
    }

    ~TestFixture()
    {
    }

    void test_solve(mat weights, float tolerance = 0.001)
    {
        mat signal = sum_signal(weights, m_signals);
        mat result = m_solver.solve(signal);
        BOOST_TEST_MESSAGE(boost::unit_test::framework::current_test_case().p_name);
        BOOST_TEST_MESSAGE("Expected: " << weights);
        BOOST_TEST_MESSAGE("Actual: " << result);
        BOOST_CHECK(is_equal(weights, result, tolerance));
    }
};