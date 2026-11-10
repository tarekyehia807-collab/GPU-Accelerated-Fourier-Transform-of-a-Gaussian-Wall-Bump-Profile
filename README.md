# GPU-Accelerated Fourier Transform of a Gaussian Wall-Bump Profile

## Overview

This project implements and benchmarks a CUDA-accelerated numerical evaluation of the Fourier transform of a Gaussian wall-bump profile.

The mathematical formulation is based on the Gaussian wall-bump representation discussed in **Page & Zaki, Journal of Fluid Mechanics, Volume 801 (2016), Section 5.1, Equation (5.1)**.

The goal is to evaluate the Fourier transform numerically on a GPU, compare its performance against a serial CPU implementation, and validate both implementations against the analytical solution.

This project focuses on one numerical operation. It does not implement the flow solver or reproduce the full physical simulation presented in the reference paper.

## Mathematical Formulation

The Gaussian wall-bump profile is

$$
H'(x)=h\exp\left(-\frac{x^2}{l_x^2}\right).
$$

Its Fourier transform is defined as

$$
H(k)=\int_{-\infty}^{+\infty}H'(x)e^{-ikx}\,dx.
$$

Using the parameters

$$
h=1,\qquad l_x=1,
$$

the transform becomes

$$
H(k)=\int_{-\infty}^{+\infty}e^{-x^2}e^{-ikx}\,dx.
$$

Since the Gaussian profile is even, the imaginary component vanishes, allowing the integral to be evaluated using symmetry:

$$
H(k)=2\int_0^\infty e^{-x^2}\cos(kx)\,dx.
$$

The analytical solution used for validation is

$$
\boxed{
H_{\mathrm{exact}}(k)=\sqrt{\pi}\exp\left(-\frac{k^2}{4}\right)
}
$$

## Numerical Method

The infinite integration domain is truncated to

$$
0\leq x\leq x_{\max},\qquad x_{\max}=8.
$$

A composite trapezoidal rule is used:

$$
H(k_m)\approx
2\Delta x
\left[
\frac{f_m(x_0)}{2}
+\sum_{j=1}^{N_x-1}f_m(x_j)
+\frac{f_m(x_{N_x})}{2}
\right],
$$

where

$$
f_m(x_j)=e^{-x_j^2}\cos(k_mx_j),
$$

and

$$
\Delta x=\frac{x_{\max}}{N_x}.
$$

The implementation evaluates the transform for uniformly spaced wavenumbers over the interval $[-10,10]$.

## CUDA Implementation

The GPU implementation uses the following strategy:

- One CUDA block computes one Fourier coefficient.
- Each block contains 256 threads.
- Threads process separate integration nodes using a strided loop.
- Each thread accumulates a local partial sum.
- Partial sums are combined through a shared-memory tree reduction.
- Both single-precision (FP32) and double-precision (FP64) implementations are evaluated.

The CPU implementation uses a serial loop over wavenumbers and integration nodes.

## Configuration

| Parameter | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| Compute capability | 8.9 |
| Number of wavenumbers | 4,096 |
| Integration intervals | 4,096 |
| Integration nodes | 4,097 |
| Wavenumber range | [-10, 10] |
| Integration upper limit | 8 |
| CUDA threads per block | 256 |
| CUDA blocks | 4,096 |
| Timing repetitions | 5 |

## Benchmark Results

The following are representative results from one execution on the hardware listed above. Actual timings may vary with GPU clocks, thermal conditions, background activity, and system load.

### FP32

| Metric | Result |
|---|---:|
| CPU serial time | 78.4136 ms |
| CUDA kernel time | 0.3815 ms |
| CUDA end-to-end time | 0.3517 ms |
| CPU / CUDA kernel speedup | 205.53× |
| CPU / CUDA end-to-end speedup | 222.93× |
| CPU maximum absolute error | 1.13 × 10⁻⁵ |
| CUDA maximum absolute error | 1.56 × 10⁻⁷ |
| CPU relative L2 error | 2.18 × 10⁻⁶ |
| CUDA relative L2 error | 4.75 × 10⁻⁸ |

### FP64

| Metric | Result |
|---|---:|
| CPU serial time | 100.8305 ms |
| CUDA kernel time | 9.9526 ms |
| CUDA end-to-end time | 10.6959 ms |
| CPU / CUDA kernel speedup | 10.13× |
| CPU / CUDA end-to-end speedup | 9.43× |
| CPU maximum absolute error | 1.64 × 10⁻¹⁴ |
| CUDA maximum absolute error | 4.44 × 10⁻¹⁶ |
| CPU relative L2 error | 3.32 × 10⁻¹⁵ |
| CUDA relative L2 error | 1.32 × 10⁻¹⁶ |

**Timing note:** Kernel-only timing and end-to-end timing use different measurement methods. Small inconsistencies between them can occur due to measurement variability; the reported values should be interpreted as measurements from an individual run, not universal performance guarantees.

## Validation

Both implementations are compared against the analytical Fourier transform.

The reported error metrics include:

- Maximum absolute error.
- Relative L2 error.
- Maximum CPU/GPU absolute difference.
- Maximum relative error over coefficients satisfying $|H(k)|>10^{-3}$.

The FP64 CUDA implementation achieves a maximum absolute error on the order of $10^{-16}$, while the FP32 implementation achieves a maximum absolute error on the order of $10^{-7}$.

## Output

The program generates a CSV file:

`gaussian_fourier_results.csv`

It contains the wavenumbers, analytical Fourier coefficients, and CPU/GPU numerical results for both FP32 and FP64.

## Building

The project can be built using NVIDIA CUDA Toolkit and a compatible Microsoft Visual Studio C++ toolchain.

For a Windows Developer Command Prompt with `nvcc` available, an example command targeting the RTX 4050 architecture is:

```bash
nvcc -O3 -std=c++17 -arch=sm_89 gaussian_wall_bump_fourier.cu -o gaussian_wall_bump_fourier.exe
```

Run the executable:

```bash
gaussian_wall_bump_fourier.exe
```

Optional command-line arguments:

```text
--nk N
--nx N
--kmax VALUE
--xmax VALUE
--repeats N
--csv PATH
--help
```

## Reference

Page, J., & Zaki, T. A. (2016). [The dynamics of spanwise vorticity perturbations in turbulent boundary layers](https://doi.org/10.1017/jfm.2016.307). *Journal of Fluid Mechanics*, 801.

The Gaussian wall-bump formulation is used here as the mathematical basis for a focused GPU numerical-computing exercise.
