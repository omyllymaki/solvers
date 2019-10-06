# Solvers

Simple practice project that implements few solvers.


## Description

The project implements few simple solvers to solve equations of the form

f(x, L) = s

where

- s = vector of observations
- L = matrix of fixed coeffients
- x = vector of unknown coefficients that needs to be solved


## Implemented solvers

**LS solver**

Assumes that f is linear (xL = s). Solves x using using least squares method.

**NNLS solver**

Assumes that f is linear (xL = s). Solves x using least squares method and constaints fit so that all x values are non-negative.

**GD solvers**

Solves f(x,L) = s using Gradient descent as optimization method. Function f is given by user.

**GN solvers**

Solves f(x,L) = s using Gauss-Newton as optimization method. Function f is given by user.

**EA solvers**

Solves f(x,L) = s using evolutionary algorithm as optimization method. Function f is given by user.


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
