
# GPU4S: Installation Guide
This document describes how to install all dependencies required to build GPU4S. For general information about the project, check the main [README](../README.md).

## Global Requirements
### 1. GNU Compiler Collection (GCC) / CLang
In order to compile the files of the project you need to have installed [**GCC**](https://gcc.gnu.org/releases.html) ≥ 7. or [**Clang**](https://releases.llvm.org/download.html) ≥ 6 in order to support C++17. 
```bash
# Install on Fedora
sudo dnf install gcc-c++ clang

# Install on Ubuntu
sudo apt update
sudo apt install build-essential clang

# Check version
g++ --version
clang --version
```


### 2. CMake
In order to build the project you need to install [**CMake**](https://cmake.org/download) ≥ 3.24 
```bash
  # Install on Fedora
  sudo dnf install cmake
  # Install on Ubuntu
  sudo apt install cmake
  # Check version
  cmake --version
```

### 3. Git

To fetch external dependencies and CMake `FetchContent` modules, you need [**Git**](https://git-scm.com/install/).

```bash
# Install on Fedora
sudo dnf install git

# Install on Ubuntu
sudo apt install git

# Check version
git --version
```
<br>

## Specific Framework Requirements

### 1. NVIDIA CUDA (nvcc)

Required for compiling [**CUDA**](https://developer.nvidia.com/cuda-downloads) accelerated benchmarks (`.cu` targets).

>Tested with version: **V13.2.86**

```bash
# Install on Fedora (via RPM Fusion / NVIDIA repo)
sudo dnf install xorg-x11-drv-nvidia-cuda cuda-toolkit

# Install on Ubuntu
sudo apt update
sudo apt install nvidia-cuda-toolkit

# Check version
nvcc --version
```
> **Additional lib NVIDIA cuDNN**<br>
> NVIDIA CUDA Deep Neural Network library (cuDNN) is required by some of the benchmark. <br>
> Tested with **V9.23.2**, can be downloaded from [NVIDIA website](https://developer.nvidia.com/cudnn-9-23-2-download-archive).

<details>
<summary><b style="font-size: 1.25em;">Installation Details</b></summary>

When downloading the local package installer from the NVIDIA website, select your operating system distribution:
* **Fedora:** Select **Linux** $\rightarrow$ **x86_64** $\rightarrow$ **RHEL** (`.rpm` package) $\rightarrow$  **FULL**.
* **Ubuntu:** Select **Linux** $\rightarrow$ **x86_64** $\rightarrow$ **Ubuntu** or **Debian** (`.deb` package) $\rightarrow$  **FULL**.


### Fedora Installation

```bash
# Download the RPM repository package for cuDNN 9.23.2
wget https://developer.download.nvidia.com/compute/cudnn/9.23.2/local_installers/cudnn-local-repo-rhel10-9.23.2-1.0-1.x86_64.rpm
# Install the package to register it with the package manager
sudo rpm -i cudnn-local-repo-rhel10-9.23.2-1.0-1.x86_64.rpm
# Clean the DNF package manager 
sudo dnf clean all

# Install the cuDNN 9 library for CUDA 13
sudo dnf -y install cudnn9-cuda-13
```

### Ubuntu Installation

```bash
# Download the local DEB package for cuDNN 9.23.2
wget https://developer.download.nvidia.com/compute/cudnn/9.23.2/local_installers/cudnn-local-repo-ubuntu2404-9.23.2_1.0-1_amd64.deb
# Install the local package to register it with APT
sudo dpkg -i cudnn-local-repo-ubuntu2404-9.23.2_1.0-1_amd64.deb
# Copy the GPG key to authenticate packages
sudo cp /var/cudnn-local-repo-ubuntu2404-9.23.2/cudnn-*-keyring.gpg /usr/share/keyrings/
# Refresh the APT package
sudo apt-get update

# Install the cuDNN 9 library for CUDA 13
sudo apt-get -y install cudnn9-cuda-13
```
</details>


### 2. AMD ROCm (hipcc)

Required for compiling [AMD HIP](https://rocm.docs.amd.com/en/latest/install/rocm.html?fam=all&w=graphics&os=ubuntu&ubuntu-ver=26.04) benchmarks (.cpp HIP targets).

>Tested with version: **6.4.43484-9999**

```Bash
# Install on Fedora
sudo dnf install rocm-hip-devel rocm-runtime

# Install on Ubuntu
sudo apt update
sudo apt install hipcc rocm-dev

# Check version
hipcc --version
```

### 3. OpenCL

Required for compiling [**OpenCL**](https://www.khronos.org/opencl/) accelerated benchmarks (`.cpp` OpenCL targets).

> Tested with version: **OpenCL 3.0**

```bash
# Install on Fedora
sudo dnf install opencl-headers ocl-icd-devel clinfo

# Install on Ubuntu
sudo apt update
sudo apt install opencl-headers ocl-icd-opencl-dev clinfo

# Check version and available devices
clinfo
```

> **Additional lib CLBlast**<br>
> The tuned OpenCL BLAS library (CLBlast) is required by some benchmarks. <br>
> Tested with **v1.6.3 and v1.7.0**, can be downloaded from the [CLBlast GitHub Releases](https://github.com/CNugteren/CLBlast/releases).

<details>
<summary><b style="font-size: 1.25em;">Installation Details</b></summary>

### Fedora Installation

```bash
# Install CLBlast development libraries directly via DNF
sudo dnf -y install clblast-devel
```

### Ubuntu Installation

```bash
# Install CLBlast development libraries directly via APT
sudo apt-get -y install libclblast-dev
```
</details>


### 4. OpenMP

Required for multi-threaded CPU parallel execution (`-fopenmp`). 

> Tested with: **OpenMP 4.5 (201511)**

```bash
# Install runtime & development libraries on Fedora
sudo dnf install libgomp

# Install runtime & development libraries on Ubuntu
sudo apt install libomp-dev

# Check supported OpenMP version via compiler macro (_OPENMP)
echo | g++ -fopenmp -dM -E - | grep _OPENMP
```
>**Additional lib OpenBLAS**<br>
> An optimized Basic Linear Algebra Subprograms (BLAS) library required by some benchmarks.<br>
> Tested with **v0.3.x**, can be downloaded from the [OpenBLAS GitHub Releases](https://github.com/OpenMathLib/OpenBLAS/releases).

<details>
<summary><b style="font-size: 1.25em;">Installation Details</b></summary>

### Fedora Installation

```bash
# Install OpenBLAS development libraries directly via DNF
sudo dnf -y install openblas-devel
```

### Ubuntu Installation

```bash
# Install OpenBLAS development libraries directly via APT
sudo apt-get -y install libopenblas-dev
```
</details>




### 5.  **Additional lib FFTW3**<br>
> A C subroutine library for computing the Discrete Fourier Transform (DFT) required by some benchmarks.<br>
> Tested with v3.3.10, can be downloaded from the [FFTW Official Website](http://www.fftw.org/download.html).

<details>
<summary><b style="font-size: 1.25em;">Installation Details</b></summary>

### Fedora Installation

```bash
# Install FFTW3 development libraries directly via DNF
sudo dnf -y install fftw-devel
```

### Ubuntu Installation

```bash
# Install FFTW3 development libraries directly via APT
sudo apt-get -y install libfftw3-dev
```
</details>

<br>

## ANDROID specific Requirements

### 1. Android NDK r27d (27.3.13750724)

First, you will need to download the [Android NDK](https://github.com/android/ndk/wiki) in your environment to cross-compile for Android targets.

You can use other versions of the Android NDK, but the project was built and successfully tested using **NDK r27d (27.3.13750724)**.

**Recommended installation instructions:**
1. Download the [Android Command Line Tools](https://developer.android.com/studio#command-line-tools-only)
2. Install them and set `ANDROID_HOME` and `sdkmanager` to your bash path:
```bash
   # Put default Android Studio path 
   # or wherever you installed it
   echo 'export ANDROID_HOME=$HOME/Android/Sdk' >> ~/.bashrc  
   echo 'export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin' >> ~/.bashrc
   source ~/.bashrc #reload bash configuration
```
3. Download your preferred version of the NDK using the `sdkmanager`:
```bash
sdkmanager --install "ndk;27.3.13750724"
```
### 2. Android Debug Bridge (ADB)

In addition, if you want to push the binaries to your phone and execute them, you should install **ADB (Android Debug Bridge)**. 

```bash
# Ubuntu 
  sudo apt install android-tools-adb
# Fedora
  sudo dnf install android-tools
```

### 3. Additional Android Libs

Some benchmarks on Android will require several pre-compiled static libraries, headers, and an OpenCL stub.

In release versions, these dependencies are already pre-compiled and included under `./gpu4s_benchmark/common/android/` this version targets the following specifications:

- Target ABIs: arm64-v8a, armeabi-v7a
-  Minimum Android API: 21
- Built With: Android NDK r27d

If you need to target a newer architecture (like ARMv9) or running the main repo, you can use my custom toolchains to download, compile the stub, libraries and headers:
- **[clblast-android-toolchain](https://github.com/EmbeddedFrime/clblast-android-toolchain)** (`libclblast.a` and `libOpenCL.so` stub)
- **[openblas-android-toolchain](https://github.com/EmbeddedFrime/openblas-android-toolchain)** (`cblas.h,` `openblas_config.h`, and `libopenblas.a`)
- **[fftw-android-toolchain](https://github.com/EmbeddedFrime/fftw-android-toolchain)** (`libfftw3.a` and `libfftw3_omp.a`)

After having recompiled the libraries, you need to replace the corresponding files in the common/android/ directory with the new ones:

<pre>
common/android/
├── include/
│   ├── arm64-v8a/
│   │   ├── cblas.h <span style="color:red">*</span>
│   │   └── openblas_config.h <span style="color:red">*</span>
│   └── armeabi-v7a/
│       ├── cblas.h <span style="color:red">*</span>
│       └── openblas_config.h <span style="color:red">*</span>
└── libs/
    ├── arm64-v8a/
    │   ├── libOpenCL.so <span style="color:blue">*</span>
    │   ├── libclblast.a <span style="color:blue">*</span>
    │   ├── libfftw3_omp.a <span style="color:green">*</span>
    │   ├── libfftw3.a <span style="color:green">*</span>
    │   └── libopenblas.a <span style="color:red">*</span>
    └── armeabi-v7a/
        ├── libOpenCL.so <span style="color:blue">*</span>
        ├── libopenblas.a <span style="color:blue">*</span>
        ├── libfftw3_omp.a <span style="color:green">*</span>
        ├── libfftw3.a <span style="color:green">*</span>
        └── libopenblas.a <span style="color:red">*</span>

OpenBLAS toolchain<span style="color:red">*</span>  CLBlast toolchain<span style="color:blue">*</span>  fftw3 toolchain<span style="color:green">*</span>
</pre>
