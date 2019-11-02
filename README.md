# Solvers

The project implements few solvers to solve equations of the form

f(x, L) = s

where

s = vector of observations (signal)

L = matrix of fixed coeffients (library)

x = vector of unknown coefficients that needs to be solved (weights)


## Implemented solvers

**LS solvers**

Assumes that f is linear (xL = s). Solves x using using least squares method.

**GD solvers**

Solves f(x,L) = s using Gradient descent as optimization method. Function f is given by user.

**GN solvers**

Solves f(x,L) = s using Gauss-Newton as optimization method. Function f is given by user.

**EA solvers**

Solves f(x,L) = s using evolutionary algorithm as optimization method. Function f is given by user.

**NN solvers**

Solves f(x,L) = s by constraining fit so that all x values are non-negative.

**Robust solvers**

Solves f(x,L) = s using robust fit that can handle considerable amount of outliers.


## Dependencies

Required: armadillo, boost

Optional: matplotlib-cpp, Python 2.7 (for plotting)

Installation:
```
$ ./scripts/install_libraries.sh
```


## Build

```
$ ./scripts/make_build.sh       (release)
$ ./scripts/make_build.sh -d    (debug)
```

## Testing

In build/tests directory run

```
$ ctest --verbose
```

## Usage

**Basic usage**

Initialize solver you want use, e.g. Gauss-Newton solver. Then solve x using solve method by passing signal as argument.

```
auto solver = GNSolver(L);
auto solution = solver.solve(s);
```

**Combination solver**

Some of the functionalities can be combined together. We can make e.g. robust RANSAC solver that uses gradient descent for optimization.

```
auto solver = GDSolver(L);
auto ransac_solver = RansacSolver<GDSolver>(solver, n_channels, accepted_error, n_accepted_points);
```
  
**Setting model**

All the solvers use linear model by default. Numerical solver can use different model specified by user. We can make e.g. Gauss-Newton solver that uses quadratic model.

```
arma::mat quadratic_model(arma::mat x, arma::mat L)
{
    return x * arma::pow(L, 2);
}

auto gn_solver = GNSolver(L);
gn_solver.set_model(quadratic_model);
```

**More information**

For more information, see examples in samples folder.
