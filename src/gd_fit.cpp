#include "gd_fit.h"
#include <iostream>

using arma::as_scalar;
using arma::mat;
using arma::randn;
using namespace std;

mat rmse(mat estimate, mat expected)
{
    mat residual = estimate - expected;
    return sqrt(sum(pow(residual, 2), 1) / residual.n_elem);
}

mat linear_model(mat x, mat L)
{
    return x * L;
}

mat gd_fit(const mat &L,
           const mat &s,
           const double &lr,
           const int &max_iter,
           const double &termination_threshold,
           mat f_model(mat, mat),
           mat f_objective(mat, mat))
{
    /*
    Solves f_model(x, L) = s using f_objective as objective function (function that will be minimized). 
    
    Solution is found using gradient decent as optimization method.

    f_model and f_objective are specified by user. By default, f_model is linear model and f_objective is RMSE.
     */
    const double x_delta = pow(10, -6);
    mat x = randn(1, L.n_rows);
    mat objective_prev = {pow(10, 16)};

    for (size_t n = 0; n < max_iter; n++)
    {

        // Calculate current value of objective
        mat s_estimate = f_model(x, L);
        mat objective0 = f_objective(s_estimate, s);

        // Calculate numerical gradient at point x
        mat gradient;
        for (size_t i = 0; i < x.n_elem; i++)
        {
            mat xt = x;
            xt[i] = x[i] + x_delta;
            s_estimate = f_model(xt, L);

            mat objective = f_objective(s_estimate, s);

            mat derivative = (objective - objective0) / x_delta;
            gradient.insert_cols(i, derivative);
        }

        // Update solution using gradient decent
        mat x_step = lr * objective0 * gradient;
        x = x - x_step;

        // Check termination condition
        mat rel_obj_change = (objective_prev - objective0) / objective0;
        if (as_scalar(rel_obj_change) < termination_threshold)
        {
            cout << "Change in objective value smaller than specified threshold" << endl;
            cout << "Iteration terminated at round " << n << endl;
            return x;
        }
        objective_prev = objective0;
    }

    cout << "Maximum number of iterations was reached" << endl;
    return x;
}