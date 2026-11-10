#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

    constexpr double kPi = 3.141592653589793238462643383279502884;
    constexpr int kThreadsPerBlock = 256;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t error__ = (call);                                     \
        if (error__ != cudaSuccess) {                                           \
            std::ostringstream message__;                                      \
            message__ << "CUDA error at " << __FILE__ << ':' << __LINE__       \
                      << ": " << cudaGetErrorString(error__);                  \
            throw std::runtime_error(message__.str());                         \
        }                                                                       \
    } while (false)

    struct Options {
        std::size_t nk = 4096;       // Number of Fourier wavenumbers.
        std::size_t nx = 4096;       // Number of trapezoidal intervals
        int repeats = 5;
        double kmax = 10.0;
        double xmax = 8.0;
        std::string csv_path = "gaussian_fourier_results.csv";
    };

    struct ErrorStats {
        double max_abs = 0.0;
        double rms_abs = 0.0;
        double relative_l2 = 0.0;
        double max_rel_resolved = 0.0;
        double max_cpu_gpu_abs = 0.0;
    };

    struct PrecisionOutput {
        std::vector<double> cpu_as_double;
        std::vector<double> gpu_as_double;
        ErrorStats cpu_error;
        ErrorStats gpu_error;
        double cpu_ms = 0.0;
        double gpu_kernel_ms = 0.0;
        double gpu_end_to_end_ms = 0.0;
    };

    void print_usage(const char* executable) {
        std::cout
            << "Usage: " << executable << " [options]\n"
            << "  --nk N         Number of uniformly spaced Fourier wavenumbers (default 4096)\n"
            << "  --nx N         Number of trapezoidal intervals; nodes = N+1 (default 4096)\n"
            << "  --kmax VALUE   Wavenumber range is [-kmax, +kmax] (default 10)\n"
            << "  --xmax VALUE   Integrate from x=0 to xmax (default 8)\n"
            << "  --repeats N    Timing repetitions (default 5)\n"
            << "  --csv PATH     Output CSV path (default gaussian_fourier_results.csv)\n"
            << "  --help         Show this message\n";
    }

    std::size_t parse_size(const std::string& value, const char* option_name) {
        std::size_t consumed = 0;
        unsigned long long parsed = 0;
        try {
            parsed = std::stoull(value, &consumed);
        }
        catch (...) {
            throw std::invalid_argument(std::string("Invalid value for ") + option_name + ": " + value);
        }
        if (consumed != value.size() || parsed == 0) {
            throw std::invalid_argument(std::string("Invalid positive integer for ") + option_name + ": " + value);
        }
        return static_cast<std::size_t>(parsed);
    }

    double parse_positive_double(const std::string& value, const char* option_name) {
        std::size_t consumed = 0;
        double parsed = 0.0;
        try {
            parsed = std::stod(value, &consumed);
        }
        catch (...) {
            throw std::invalid_argument(std::string("Invalid value for ") + option_name + ": " + value);
        }
        if (consumed != value.size() || !std::isfinite(parsed) || parsed <= 0.0) {
            throw std::invalid_argument(std::string("Expected a finite positive value for ") + option_name);
        }
        return parsed;
    }

    Options parse_options(int argc, char** argv) {
        Options options;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help") {
                print_usage(argv[0]);
                std::exit(EXIT_SUCCESS);
            }
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing value after " + arg);
            }
            const std::string value = argv[++i];
            if (arg == "--nk") {
                options.nk = parse_size(value, "--nk");
            }
            else if (arg == "--nx") {
                options.nx = parse_size(value, "--nx");
            }
            else if (arg == "--repeats") {
                const std::size_t parsed = parse_size(value, "--repeats");
                if (parsed > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                    throw std::invalid_argument("--repeats is too large");
                }
                options.repeats = static_cast<int>(parsed);
            }
            else if (arg == "--kmax") {
                options.kmax = parse_positive_double(value, "--kmax");
            }
            else if (arg == "--xmax") {
                options.xmax = parse_positive_double(value, "--xmax");
            }
            else if (arg == "--csv") {
                options.csv_path = value;
                if (options.csv_path.empty()) {
                    throw std::invalid_argument("--csv path cannot be empty");
                }
            }
            else {
                throw std::invalid_argument("Unknown option: " + arg);
            }
        }

        if (options.nk > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
            throw std::invalid_argument("--nk exceeds the CUDA 1D grid limit used by this program");
        }
        if (options.nx > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::invalid_argument("--nx is too large");
        }
        return options;
    }

    // Composite trapezoidal rule for
    // H(k) = 2 * integral_0^xmax exp(-x^2) cos(k*x) dx.
    // nx is the number of intervals, so the sum contains nx+1 nodes.
    template <typename T>
    void compute_cpu(const std::vector<T>& wavenumbers,
        std::size_t nx,
        T xmax,
        std::vector<T>& output) {
        const T dx = xmax / static_cast<T>(nx);
        const std::size_t nk = wavenumbers.size();

        for (std::size_t m = 0; m < nk; ++m) {
            const T k = wavenumbers[m];
            T sum = static_cast<T>(0);

            for (std::size_t j = 0; j <= nx; ++j) {
                const T x = static_cast<T>(j) * dx;
                const T endpoint_weight = (j == 0 || j == nx)
                    ? static_cast<T>(0.5)
                    : static_cast<T>(1.0);
                const T integrand = std::exp(-x * x) * std::cos(k * x);
                sum += endpoint_weight * integrand;
            }
            output[m] = static_cast<T>(2) * dx * sum;
        }
    }

    // One CUDA block computes one Fourier coefficient. Threads in the block
    // integrate disjoint x-nodes, then reduce their partial sums in shared memory.
    template <typename T>
    __global__ void gaussian_fourier_kernel(const T* __restrict__ wavenumbers,
        T* __restrict__ output,
        std::size_t nk,
        std::size_t nx,
        T xmax) {
        extern __shared__ unsigned char shared_bytes[];
        T* partial_sums = reinterpret_cast<T*>(shared_bytes);

        const unsigned int tid = threadIdx.x;
        const std::size_t mode = static_cast<std::size_t>(blockIdx.x);
        if (mode >= nk) {
            return;
        }

        const T k = wavenumbers[mode];
        const T dx = xmax / static_cast<T>(nx);
        T local_sum = static_cast<T>(0);

        for (std::size_t j = static_cast<std::size_t>(tid); j <= nx; j += blockDim.x) {
            const T x = static_cast<T>(j) * dx;
            const T endpoint_weight = (j == 0 || j == nx)
                ? static_cast<T>(0.5)
                : static_cast<T>(1.0);
            const T integrand = exp(-x * x) * cos(k * x);
            local_sum += endpoint_weight * integrand;
        }

        partial_sums[tid] = local_sum;
        __syncthreads();

        // blockDim.x is 256 (a power of two), so this tree reduction is valid.
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partial_sums[tid] += partial_sums[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            output[mode] = static_cast<T>(2) * dx * partial_sums[0];
        }
    }

    template <typename T>
    void launch_kernel(const T* d_wavenumbers,
        T* d_output,
        std::size_t nk,
        std::size_t nx,
        T xmax) {
        const std::size_t shared_memory_bytes =
            static_cast<std::size_t>(kThreadsPerBlock) * sizeof(T);

        gaussian_fourier_kernel<T> << <static_cast<unsigned int>(nk),
            kThreadsPerBlock,
            shared_memory_bytes >> > (
                d_wavenumbers, d_output, nk, nx, xmax);
        CUDA_CHECK(cudaGetLastError());
    }

    template <typename T>
    ErrorStats calculate_error(const std::vector<T>& numerical,
        const std::vector<T>& other_numerical,
        const std::vector<T>& wavenumbers) {
        ErrorStats stats;
        long double sum_error_squared = 0.0L;
        long double sum_exact_squared = 0.0L;
        long double sum_absolute_error_squared = 0.0L;

        for (std::size_t i = 0; i < numerical.size(); ++i) {
            const double k = static_cast<double>(wavenumbers[i]);
            const double exact = std::sqrt(kPi) * std::exp(-0.25 * k * k);
            const double value = static_cast<double>(numerical[i]);
            const double other = static_cast<double>(other_numerical[i]);
            const double abs_error = std::abs(value - exact);
            const double cpu_gpu_error = std::abs(value - other);

            stats.max_abs = std::max(stats.max_abs, abs_error);
            stats.max_cpu_gpu_abs = std::max(stats.max_cpu_gpu_abs, cpu_gpu_error);
            sum_error_squared += static_cast<long double>(abs_error) * abs_error;
            sum_exact_squared += static_cast<long double>(exact) * exact;
            sum_absolute_error_squared += static_cast<long double>(abs_error) * abs_error;

            // Relative error is reported only where the exact coefficient is not
            // vanishingly small; otherwise cancellation dominates the ratio.
            if (std::abs(exact) > 1.0e-3) {
                stats.max_rel_resolved = std::max(stats.max_rel_resolved, abs_error / std::abs(exact));
            }
        }

        const long double count = static_cast<long double>(numerical.size());
        stats.rms_abs = std::sqrt(static_cast<double>(sum_absolute_error_squared / count));
        stats.relative_l2 = (sum_exact_squared > 0.0L)
            ? std::sqrt(static_cast<double>(sum_error_squared / sum_exact_squared))
            : 0.0;
        return stats;
    }

    template <typename T>
    PrecisionOutput run_precision(const char* precision_name,
        const std::vector<T>& wavenumbers,
        const Options& options) {
        PrecisionOutput result;
        const std::size_t nk = wavenumbers.size();
        const T xmax = static_cast<T>(options.xmax);

        std::vector<T> cpu_output(nk, static_cast<T>(0));
        std::vector<T> gpu_output(nk, static_cast<T>(0));

        // CPU reference and timing.
        compute_cpu(wavenumbers, options.nx, xmax, cpu_output); // warm-up / correctness result
        const auto cpu_begin = std::chrono::steady_clock::now();
        for (int repeat = 0; repeat < options.repeats; ++repeat) {
            compute_cpu(wavenumbers, options.nx, xmax, cpu_output);
        }
        const auto cpu_end = std::chrono::steady_clock::now();
        result.cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_begin).count()
            / static_cast<double>(options.repeats);

        T* d_wavenumbers = nullptr;
        T* d_output = nullptr;
        const std::size_t input_bytes = nk * sizeof(T);
        const std::size_t output_bytes = nk * sizeof(T);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_wavenumbers), input_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output), output_bytes));

        try {
            CUDA_CHECK(cudaMemcpy(d_wavenumbers, wavenumbers.data(), input_bytes, cudaMemcpyHostToDevice));

            // Warm-up launches are excluded from event timing.
            launch_kernel(d_wavenumbers, d_output, nk, options.nx, xmax);
            launch_kernel(d_wavenumbers, d_output, nk, options.nx, xmax);
            CUDA_CHECK(cudaDeviceSynchronize());

            // One correctness output, copied back after warm-up.
            CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));

            // Kernel-only timing: input is already resident on the GPU.
            cudaEvent_t start_event = nullptr;
            cudaEvent_t stop_event = nullptr;
            CUDA_CHECK(cudaEventCreate(&start_event));
            CUDA_CHECK(cudaEventCreate(&stop_event));
            CUDA_CHECK(cudaEventRecord(start_event));
            for (int repeat = 0; repeat < options.repeats; ++repeat) {
                launch_kernel(d_wavenumbers, d_output, nk, options.nx, xmax);
            }
            CUDA_CHECK(cudaEventRecord(stop_event));
            CUDA_CHECK(cudaEventSynchronize(stop_event));
            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
            result.gpu_kernel_ms = static_cast<double>(elapsed_ms) / options.repeats;
            CUDA_CHECK(cudaEventDestroy(start_event));
            CUDA_CHECK(cudaEventDestroy(stop_event));

            // End-to-end timing includes H2D copy, kernel launch/execution and D2H copy.
            const auto gpu_begin = std::chrono::steady_clock::now();
            for (int repeat = 0; repeat < options.repeats; ++repeat) {
                CUDA_CHECK(cudaMemcpy(d_wavenumbers, wavenumbers.data(), input_bytes, cudaMemcpyHostToDevice));
                launch_kernel(d_wavenumbers, d_output, nk, options.nx, xmax);
                CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));
            }
            const auto gpu_end = std::chrono::steady_clock::now();
            result.gpu_end_to_end_ms =
                std::chrono::duration<double, std::milli>(gpu_end - gpu_begin).count()
                / static_cast<double>(options.repeats);

        }
        catch (...) {
            cudaFree(d_wavenumbers);
            cudaFree(d_output);
            throw;
        }

        CUDA_CHECK(cudaFree(d_wavenumbers));
        CUDA_CHECK(cudaFree(d_output));

        result.cpu_error = calculate_error(cpu_output, gpu_output, wavenumbers);
        result.gpu_error = calculate_error(gpu_output, cpu_output, wavenumbers);

        result.cpu_as_double.resize(nk);
        result.gpu_as_double.resize(nk);
        for (std::size_t i = 0; i < nk; ++i) {
            result.cpu_as_double[i] = static_cast<double>(cpu_output[i]);
            result.gpu_as_double[i] = static_cast<double>(gpu_output[i]);
        }

        std::cout << "\n=== " << precision_name << " ===\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "CPU serial average       : " << result.cpu_ms << " ms\n";
        std::cout << "CUDA kernel average      : " << result.gpu_kernel_ms << " ms\n";
        std::cout << "CUDA end-to-end average  : " << result.gpu_end_to_end_ms << " ms\n";
        if (result.gpu_kernel_ms > 0.0) {
            std::cout << "CPU / CUDA kernel speedup: " << result.cpu_ms / result.gpu_kernel_ms << "x\n";
        }
        if (result.gpu_end_to_end_ms > 0.0) {
            std::cout << "CPU / CUDA full speedup  : " << result.cpu_ms / result.gpu_end_to_end_ms << "x\n";
        }
        std::cout << std::scientific << std::setprecision(8);
        std::cout << "CPU max absolute error   : " << result.cpu_error.max_abs << "\n";
        std::cout << "CUDA max absolute error  : " << result.gpu_error.max_abs << "\n";
        std::cout << "CPU relative L2 error    : " << result.cpu_error.relative_l2 << "\n";
        std::cout << "CUDA relative L2 error   : " << result.gpu_error.relative_l2 << "\n";
        std::cout << "CPU/GPU max abs difference: " << result.gpu_error.max_cpu_gpu_abs << "\n";
        std::cout << "CUDA max relative error (|H|>1e-3): "
            << result.gpu_error.max_rel_resolved << "\n";

        return result;
    }

    void write_csv(const std::string& path,
        const std::vector<double>& wavenumbers_fp64,
        const std::vector<float>& wavenumbers_fp32,
        const PrecisionOutput& fp32,
        const PrecisionOutput& fp64) {
        std::ofstream file(path);
        if (!file) {
            throw std::runtime_error("Could not open CSV output file: " + path);
        }

        file << "k_fp64,k_fp32,H_exact_at_k_fp64,H_exact_at_k_fp32,"
            "H_cpu_fp32,H_cuda_fp32,H_cpu_fp64,H_cuda_fp64\n";
        file << std::setprecision(17);
        for (std::size_t i = 0; i < wavenumbers_fp64.size(); ++i) {
            const double k64 = wavenumbers_fp64[i];
            const double k32_as_double = static_cast<double>(wavenumbers_fp32[i]);
            const double exact64 = std::sqrt(kPi) * std::exp(-0.25 * k64 * k64);
            const double exact32 = std::sqrt(kPi) * std::exp(-0.25 * k32_as_double * k32_as_double);
            file << k64 << ',' << k32_as_double << ',' << exact64 << ',' << exact32 << ','
                << fp32.cpu_as_double[i] << ',' << fp32.gpu_as_double[i] << ','
                << fp64.cpu_as_double[i] << ',' << fp64.gpu_as_double[i] << '\n';
        }
    }

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);

        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count <= 0) {
            throw std::runtime_error("No CUDA-capable NVIDIA GPU was detected.");
        }

        CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp device_properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&device_properties, 0));

        std::cout << "CUDA Gaussian wall-bump Fourier transform\n";
        std::cout << "Reference: Page & Zaki, JFM 801 (2016), Section 5.1, Eq. (5.1)\n";
        std::cout << "GPU: " << device_properties.name << "\n";
        std::cout << "GPU compute capability: " << device_properties.major << '.'
            << device_properties.minor << "\n";
        std::cout << "h = 1, lx = 1; H(k) = integral[-inf,+inf] exp(-x^2) exp(-i*k*x) dx\n";
        std::cout << "Using symmetry: H(k) = 2 * integral[0,xmax] exp(-x^2) cos(k*x) dx\n";
        std::cout << "Nk = " << options.nk
            << ", Nx intervals = " << options.nx
            << ", integration nodes = " << (options.nx + 1)
            << ", k range = [-" << options.kmax << ", +" << options.kmax << "]"
            << ", xmax = " << options.xmax
            << ", repeats = " << options.repeats << "\n";
        std::cout << "CUDA launch: one block per k, " << kThreadsPerBlock
            << " threads per block, shared-memory reduction\n";

        std::vector<double> k64(options.nk);
        for (std::size_t i = 0; i < options.nk; ++i) {
            const double fraction = (options.nk == 1)
                ? 0.5
                : static_cast<double>(i) / static_cast<double>(options.nk - 1);
            k64[i] = -options.kmax + 2.0 * options.kmax * fraction;
        }

        std::vector<float> k32(options.nk);
        for (std::size_t i = 0; i < options.nk; ++i) {
            k32[i] = static_cast<float>(k64[i]);
        }

        const PrecisionOutput fp32 = run_precision<float>("FP32", k32, options);
        const PrecisionOutput fp64 = run_precision<double>("FP64", k64, options);

        write_csv(options.csv_path, k64, k32, fp32, fp64);
        std::cout << "\nCSV written to: " << options.csv_path << "\n";
        return EXIT_SUCCESS;
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}