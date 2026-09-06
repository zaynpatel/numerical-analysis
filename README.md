# numerical-analysis

This repository implements 24 algorithms across the following topics:

- Root finding in single variable differentiable functions
- Direct and iterative matrix methods to solve Ax=b
- Linear least squares by factoring A = QR
- Eigenvalue solving (dominant eigenpairs and all eigenvalues)
- Nonlinear optimization of scalar and vector valued functions

My primary goal for this repository was to learn and implement several numerical algorithms used in production-grade software libraries (e.g. LAPACK). In this repository you will find implementations of Newton's method for nonlinear single variable and multivariable systems, LU and Cholesky decomposition, Householder reflections, conjugate gradient, QR eigenvalue, and more.

Another goal of mine is to provide readable code to others who are interested in numerical algorithms and prefer to read code, as opposed to mathematical notation, when being introduced to this material. For this reason I have been more explicit with some of my implementations. I have made a small effort to provide vectorized implementations of some algorithms and these are prefixed as `efficient_`. For example, there is a `gaussian_elimination` and `efficient_gaussian_elimination` function. In addition, some functions contain pydocs that include mathematical details to explain the math behind the algorithm. You will find extensive pydocs in `nonlinear_optimization.py` and `eigen_singular.py`.

Note: This repository will include more algorithms from topics like polynomial interpolation, Fourier transform, and numerical methods for DiffEq's (Runga-Kutta).

My main resource for this repository is (A First Course in Numerical Methods, by Ascher & Greif)[https://epubs.siam.org/doi/book/10.1137/9780898719987].

---
*I wrote all the code, pydocs, and tests in this repository **without LLM assistance**. All errors are mind and please reach out if you find any issues.*

Contact: zaynpatelwhs@gmail.com
