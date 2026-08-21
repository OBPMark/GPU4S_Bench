# GPU4S Benchmark Suite documentation

## Introduction
GPU4S is a benchmarking suite that rely on OBPMark-Kernel to test perfomance and reliability of GPUs and multi-threaded processors for space applications. 

The benchmarking suites have been developed in order to be compatible with heterogeneous platforms:

- Computer (x86_64, x86_32) 
- Android (ARM64, ARM32)
- Nvidia Xavier/TX2 (ARM64) (currently in development)

<details>
<summary><b style="font-size: 1em;">List of tested devices</b></summary>
Up to this date, the suite has been successfully compiled and executed on the following devices:

| Platform | Operating System | CPU | GPU | Frameworks Tested |
| :--- | :--- | :--- | :--- | :--- |
| **High-end Laptop (x86_64)** | Fedora | AMD Ryzen 7 7840HS | NVIDIA GeForce RTX 4060 Laptop | CPU, OpenMP, CUDA, OpenCL, HIP |
| **Smartphone (ARM64)** | Android 5 | Qualcomm Snapdragon 810 | Adreno 430 | CPU, OpenMP, OpenCL |
</details>
<br>


The benchmark uses a couple of different programming languages, libraries and frameworks to be able to compare perfomance of the same benchmark across most of devices: 
- Standard C/C++
- CUDA
- HIP
- OpenCL
- OpenMP


## Background

Embedded GPUs have been identified by both private companies and government space agencies as a promising technology to meet the growing demands of payload processing. The GPU4S (GPU for Space) project, funded by the European Space Agency (ESA), explores the feasibility and benefits of using embedded GPUs for space workloads, and provides guidelines for their adoption in space applications.

## Benchmark List and Basic Description

For most of the benchmark suite there is a naïve, optimized and library version. The benchmarks with their implementations are listed below.

| Benchmark | Naïve | Optimized | Library |
|---|:---:|:---:|:---:|
| Cifar 10 | ✅ | ✅ | ✅ (CUDA only) |
| Cifar 10 Multiple | ✅ | ✅ | ✅ (CUDA only) |
| Convolution 2D | ✅ | ✅ | ✅ (CUDA only) |
| Correlation 2D | ✅ | ✅ | ❌ |
| Fast Fourier Transform 2D | ❌ | ❌ | ✅ |
| Fast Fourier Transform | ✅ | ✅ | ✅ |
| Fast Fourier Transform Window | ✅ | ✅ | ✅ |
| Finite Impulse Response Filter | ✅ | ❌ | ❌ |
| Local Response Normalization (LRN) | ✅ | ✅ | ✅ (CUDA only) |
| Matrix Multiplication | ✅ | ✅ | ✅ |
| Max Pooling | ✅ | ✅ | ✅ (CUDA only) |
| Memory Bandwidth | ✅ | ❌ | ❌ |
| ReLU | ✅ | ✅ | ✅ (CUDA only) |
| Softmax | ✅ | ✅ | ✅ (CUDA only) |
| Wavelet Transform | ✅ | ✅ | ❌ |

## Quick Start

If you already have the basic C/C++ programming tools installed (GCC/Clang, CMake ≥ 3.24, Git — see [**Prerequisites**](./docs/INSTALL.md) if not), you can try to compile and run the CPU version of the matrix multiplication benchmark in 3 steps:

```bash
# 1. Go to the benchmark directory
cd gpu4s_benchmark/matrix_multiplication_bench

# 2. Generate build files and compile the matrix_mult CPU target
cmake -B build
cmake --build build --target cpu -j$(nproc)

# 3. Run it
./build/bin/matrix_mult_cpu -s 512 -t
```

- `-s 512` runs the benchmark on a 512x512 matrix
- `-t` prints the execution time

Congratulations! You have successfully built and run your first GPU4S benchmark.

> Wanting to build with CUDA, HIP, OpenCL, or for Android? See [docs/INSTALL.md](docs/INSTALL.md) for prerequisites and [docs/BUILD_AND_RUN.md](docs/BUILD_AND_RUN.md) for building targets and check the runtime options.



## Road map
**Main focus**:

- [X] fix issue of correctness between cpu and gpu in some benchmarks
- [X] refactor to extract the common of frameworks
- [X] fix of the clock to be executed in runtime + kernerCLK->deviceOBJ
- [X] Check for cl error during memory copy to host and clean
- [X] add UMA implementation for Android
- [X] Big cmake to compile everything 
- [ ] be compatible with jetson board + add UMA for jetson board

**Bonus**:
- [ ] create a test with vulkan for android to have best performance (with softmax ?)
- [ ] add map for verification of the result (ANDROID UMA)
- [ ] refactor main, cuda, hip, opencl, OpenMP, -> create common 
- [ ] refactor cpu function -> create common  (most important and easier)
- [ ] add a get elapsed time function that print in this cpu function

## The Authors
- Ivan Rodriguez Ferrandez (BSC-UPC)
- Alvaro Jover-Alvarez (BSC-UPC)
- Leonidas Kosmidis (BSC-UPC)
- Noah Perret (BSC-Centrale Nantes)
- David Steenari (ESA)

## License

[ESA-PL Strong Copyleft – v2.5](./LICENSE)