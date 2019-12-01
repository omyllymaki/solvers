#include "numerical_solver.h"

void NumericalSolver::set_model(model_wrapper f_model)
{
    model = f_model;
}
void NumericalSolver::set_initial_guess(arma::mat initial_guess)
{
    m_init_guess = initial_guess;
}

void NumericalSolver::initialize_solution() {
    if (m_init_guess.is_empty())
    {
        m_x = arma::zeros(m_s.n_rows, m_L.n_rows);
    }
    else
    {
        m_x = m_init_guess;
    }
}