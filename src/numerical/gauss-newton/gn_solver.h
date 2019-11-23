#ifndef GN_SOLVER_H
#define GN_SOLVER_H

#include "../numerical_solver.h"

//! Gauss-Newton solver.
//! Uses Gauss-Newton method to find lowest RMSE for objective.
class GNSolver : public NumericalSolver
{

protected:
    double m_termination_threshold;
    static constexpr double m_x_delta = 0.000001;
    arma::mat m_objective_prev, m_objective, m_jacobian, m_residual;

    virtual void update_solution() override;

    virtual bool is_termination_condition_filled() override;

    virtual arma::mat objective(arma::mat estimate, arma::mat expected) override;

    virtual void update_jacobian();

    virtual void update_objective();

    virtual void update_residual();

public:
    GNSolver(){};

    //! Solver initialization.
    //! @param L - Library used to fit model f(x,L) = s.
    //! @param max_iter - Maximum number of iterations.
    //! @param termination_threshold - Threshold value for objective. The iteration is terminated when threshold value is reached.
    GNSolver(const arma::mat &L,
             const int max_iter = 100,
             const double termination_threshold = 0.000001);

    virtual arma::mat solve(const arma::mat &s) override;
};

#endif