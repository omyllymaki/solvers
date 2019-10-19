# Solvers

The project implements few solvers to solve equations of the form

f(x, L) = s

where

s = vector of observations (signal)

L = matrix of fixed coeffients (library)

x = vector of unknown coefficients that needs to be solved (weights)


## Implemented solvers

All the solvers can be constrained so that all x values are forced to be non-negative.

**LS solvers**

Assumes that f is linear (xL = s). Solves x using using least squares method.

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
