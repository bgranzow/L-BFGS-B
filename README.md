## L-BFGS-B
> A pure Matlab implementation of the L-BFGS-B algorithm.

### Introduction
The L-BFGS-B algorithm is a limited memory quasi-Newton,
gradient based optimzation algorithm to solve problems
of the form:
```
minimize f(x)
such that l <= x <= u
```

### Motivation
The L-BFGS-B algorithm uses a limited memory BFGS
representation of the Hessian matrix, making it well-suited
for optimization problems with a large number of design
variables. Many wrappers (C/C++, Matlab, Python, Julia) to
the [original L-BFGS-B Fortran implementation][1] exist, but a
pure Matlab implementation of the algorithm (as far as I
could tell) did not exist up to this point. This is likely due
to performance concerns. Nevertheless, this single file
implementation (`LBFGS.m`) of the L-BFGS-B algorithm
seeks to provide Matlab users a convenient way to
optimize bound-constrained problems with the L-BFGS-G
algorithm without installing third-party software.

### Background
* The [original L-BFGS-B paper][0]
* The [original L-BFGS-B fortran implementation][1]
* Numerical optimization [fundamentals][2]

[0]:http://epubs.siam.org/doi/abs/10.1137/0916069
[1]:http://users.iems.northwestern.edu/~nocedal/lbfgsb.html
[2]:http://link.springer.com/book/10.1007%2F978-0-387-40065-5
[3]:https://en.wikipedia.org/wiki/Limited-memory_BFGS
