# Solvers

Simple practice project that implements few solvers.


## Description

The project implements few simple solvers to solve equations of the form

f(x,L) = s

where

- s = vector of observations
- L = matrix of fixed coeffients
- x = vector of unknown coefficients that needs to be solved


## Implemented solvers

**LS solver**

Solves xL = s using least squares fit.

**NNLS solver**

Solves xL = s using least squares method and constaints fit so that all x values are non-negative.

**GD solvers**

Solves f(x,L) using Gradient descent optimization. Function f and objective function are implemented by individual GD solvers.


## Dependencies

Armadillo, Boost

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
