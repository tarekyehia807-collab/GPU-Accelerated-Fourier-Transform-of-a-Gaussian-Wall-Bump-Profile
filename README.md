# GPU-Accelerated Fourier Transform of a Gaussian Wall-Bump Profile

## Overview

This project implements and benchmarks a CUDA-accelerated numerical evaluation of the Fourier transform of a Gaussian wall-bump profile.

The mathematical formulation is based on Section 5.1, titled "Gaussian wall bump," in the following paper:

**Page, J., & Zaki, T. A. (2016). Viscoelastic shear flow over a wavy surface. Journal of Fluid Mechanics, 801, 392–429.**

The objective is to evaluate the Fourier transform numerically on a GPU, compare its performance against a serial CPU implementation, and validate both implementations against the analytical solution.

The project isolates the Fourier-transform operation associated with the wall geometry. It does not implement the viscoelastic flow solver or reproduce the full physical simulations presented in the paper.

## Mathematical Formulation

The project considers the nondimensional Gaussian wall-bump profile introduced in Section 5.1 of the reference paper:

$$
H'(x) =
h\exp\left(
-\frac{|x|^2}{l_x^2}
\right).
$$

Here, the prime denotes the wall-disturbance notation used in the paper; it does not indicate differentiation with respect to the spatial coordinate.

The Fourier transform of the wall disturbance is defined as:

$$
H(k) =
\int_{-\infty}^{+\infty}
H'(x)e^{-ikx}\,dx.
$$

For this implementation, the wall-bump length scale and amplitude are both set to unity:

$$
l_x=1,
\qquad
h=1.
$$

Consequently, the Gaussian profile becomes:

$$
H'(x)=e^{-x^2}.
$$

The corresponding Fourier transform is:

$$
H(k)=
\int_{-\infty}^{+\infty}
e^{-x^2}e^{-ikx}\,dx.
$$

Since the Gaussian profile is even, the imaginary component of the Fourier integral vanishes. Therefore, the transform can be evaluated using symmetry:

$$
H(k)=
2\int_0^\infty
e^{-x^2}\cos(kx)\,dx.
$$

The analytical Fourier transform is:

$$
H_{\mathrm{exact}}(k)
=
\sqrt{\pi}\exp\left(-\frac{k^2}{4}\right).
$$

This analytical expression is used as the reference solution for numerical validation.

## Numerical Method

The infinite integration domain is truncated to a finite interval:

$$
0\leq x\leq x_{\max},
\qquad
x_{\max}=8.
$$

The integral is evaluated using the composite trapezoidal rule:

$$
H(k_m)\approx
2\Delta x
\left[
\frac{f_m(x_0)}{2}
+
\sum_{j=1}^{N_x-1}f_m(x_j)
+
\frac{f_m(x_{N_x})}{2}
\right],
$$

where the integrand is:

$$
f_m(x_j)=e^{-x_j^2}\cos(k_mx_j),
$$

and the spatial grid spacing is:

$$
\Delta x=\frac{x_{\max}}{N_x}.
$$

The implementation evaluates the transform at uniformly spaced Fourier wavenumbers over the following interval:

$$
-10\leq k\leq 10.
$$

The numerical results are compared against the analytical solution at the same wavenumber values.

Each Fourier coefficient is evaluated directly through numerical quadrature rather than using an FFT.

## CPU and CUDA Implementations

### CPU Reference

The CPU implementation evaluates the integral serially:

- An outer loop iterates over the Fourier wavenumbers.
- An inner loop evaluates the integrand at the integration nodes.
- The weighted contributions are accumulated using the composite trapezoidal rule.

### CUDA Implementation

The GPU implementation uses a block-level parallel reduction:

- One CUDA block computes one Fourier coefficient.
- Each block contains 256 threads.
- Threads process separate integration nodes using a strided loop.
- Each thread accumulates a local partial sum.
- Partial sums are combined using a shared-memory tree reduction.
- The final result is written by thread 0 of each block.

Both single-precision (FP32) and double-precision (FP64) implementations are evaluated.

## Configuration

| Parameter | Value |
|---|---:|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| Compute capability | 8.9 |
| Wall-bump amplitude | 1 |
| Wall-bump length scale | 1 |
| Number of wavenumbers | 4,096 |
| Integration intervals | 4,096 |
| Integration nodes | 4,097 |
| Wavenumber range | [-10, 10] |
| Integration upper limit | 8 |
| CUDA threads per block | 256 |
| CUDA blocks | 4,096 |
| Timing repetitions | 5 |

## Benchmark Results

The following results are from one representative execution. Each reported timing is averaged over five repetitions within that execution.

Actual performance may vary depending on GPU clocks, thermal conditions, background activity, and system load.

### FP32

| Metric | Result |
|---|---:|
| CPU serial average | 78.7410 ms |
| CUDA kernel average | 0.2508 ms |
| CUDA end-to-end average | 0.4257 ms |
| CPU / CUDA kernel speedup | 313.93x |
| CPU / CUDA end-to-end speedup | 184.96x |
| CPU maximum absolute error | 1.13 x 10^-5 |
| CUDA maximum absolute error | 1.56 x 10^-7 |
| CPU relative L2 error | 2.18 x 10^-6 |
| CUDA relative L2 error | 4.75 x 10^-8 |

### FP64

| Metric | Result |
|---|---:|
| CPU serial average | 101.2969 ms |
| CUDA kernel average | 9.8493 ms |
| CUDA end-to-end average | 10.9880 ms |
| CPU / CUDA kernel speedup | 10.28x |
| CPU / CUDA end-to-end speedup | 9.22x |
| CPU maximum absolute error | 1.64 x 10^-14 |
| CUDA maximum absolute error | 4.44 x 10^-16 |
| CPU relative L2 error | 3.32 x 10^-15 |
| CUDA relative L2 error | 1.32 x 10^-16 |

**Timing methodology:**

- Kernel-only timing measures GPU kernel execution using CUDA events, with input data already resident on the GPU.
- End-to-end timing includes host-to-device transfer, kernel execution, and device-to-host transfer.
- CPU timing measures the serial computation.

The speedup values are calculated from the measured CPU and GPU times. They describe this particular workload and hardware configuration, not a universal performance guarantee.

## Numerical Validation

Both implementations are compared against the analytical Fourier transform.

The following error metrics are evaluated:

- Maximum absolute error.
- Relative L2 error.
- Maximum CPU/GPU absolute difference.
- Maximum relative error for coefficients satisfying the following condition:

$$
\left|H_{\mathrm{exact}}(k)\right|>10^{-3}.
$$

The FP64 CUDA implementation achieves a maximum absolute error on the order of 10^-16.

The FP32 CUDA implementation achieves a maximum absolute error on the order of 10^-7.

These results demonstrate agreement between the numerical implementations and the analytical Fourier transform at both precision levels.

## Output

The program generates the following file:

`gaussian_fourier_results.csv`

This allows the numerical results to be independently inspected and plotted.

## Building

The project requires:

- NVIDIA CUDA Toolkit.
- A compatible Microsoft Visual Studio C++ toolchain.
- An NVIDIA GPU supporting CUDA.

For a Windows Developer Command Prompt with `nvcc` available, an example compilation command targeting the RTX 4050 architecture is:

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

Page, J., & Zaki, T. A. (2016). **Viscoelastic shear flow over a wavy surface.** *Journal of Fluid Mechanics*, 801, 392–429.

https://doi.org/10.1017/jfm.2016.455

**Relevant section:** Section 5.1, "Gaussian wall bump."
(https://engineering.jhu.edu/zaki/wp-content/uploads/2018/12/Page_JFM2016.pdf)
