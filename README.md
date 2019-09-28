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

Solves f(x,L) = s using Gradient descent optimization where function f is given by user. Class GDSolver provides general GD solver with some default behaviour. Behaviour can be easily modified by inheriting the class and implementing corresponding protected methods. E.g. GDLinearSolver inherits GDSolver and implements custom learning rate update and objective function.


## Dependencies

Required: armadillo, boost

Optional: matplotlib-cpp

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
