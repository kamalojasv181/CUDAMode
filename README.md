# CUDA MODE

## Introduction
This is [my](https://github.com/kamalojasv181) personal practice repository for the [CUDA MODE](https://github.com/cuda-mode) course. This repository contains learning materials, code examples, and practical implementations focused on CUDA programming, GPU optimization, and parallel computing concepts.

## Repository Structure

### 📚 Books/
Contains reference materials and textbooks used in the course:
- **PMPP/** - Programming Massively Parallel Processors book and resources
  - `PMPP.pdf` - The main textbook
  - `InstructorResources/` - Lab solutions and instructor materials including:
    - Vector addition implementations
    - Matrix multiplication (SGEMM) examples
    - Convolution operations
    - Reduction and prefix scan algorithms
    - Histogram computation
    - Device query utilities

### 💻 Codes/
Organized by lecture, containing practical implementations and examples:
- **Lecture1/** - Introduction to GPU programming
  - `pytorch_square.py` - PyTorch tensor operations and profiling
  - `numba_square.py` - Numba CUDA implementations
  - `triton_square.py` - Triton kernel examples
  - `triton_square_torch_compiled.py` - Torch compiled versions
  - `load_inline.py` - Inline CUDA kernel loading
  - Performance analysis files (CSV, PNG, HTML reports)
  - `square_kernel.ptx` - Generated PTX assembly code

- **Lecture2/** - Advanced CUDA concepts
  - `PMPP/Chapter2/` - Chapter-specific implementations from the PMPP book

### 📝 HandWrittenNotes/
Personal notes and annotations from lectures:
- `Lecture1.pdf` - Handwritten notes from the first lecture

### 📊 Slides/
Presentation materials from the course:
- `Lecture_1.pptx` - First lecture slides
- `Lecture2.pptx` - Second lecture slides  
- `Lecture4.pdf` - Fourth lecture slides

### 🌐 OfficialResources/
Placeholder for official course materials:
- `lectures/` - Official lecture resources
- `resource-stream/` - Additional course resources

### 🐳 Docker Environment
- `Dockerfile` - Complete CUDA development environment setup including:
  - NVIDIA CUDA 12.3.1 with Ubuntu 22.04
  - Python 3 with essential packages (PyTorch, NumPy, Numba, Triton)
  - Development tools and dependencies

## Installation & Setup

### Using Docker (Recommended)
I use a Docker container to ensure a consistent CUDA development environment:

```bash
# Build the container
docker build -t cuda-mode .

# Run the container with GPU access
docker run --gpus all -v /home:/home --ipc=host -it -d --name cuda-mode-container cuda-mode

# Access the container
docker exec -it cuda-mode-container bash
```

### Prerequisites
- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Toolkit
- NVIDIA drivers installed on host system

## Course Topics Covered

1. **GPU Architecture & Programming Models**
2. **CUDA Kernel Development**
3. **Memory Management & Optimization**
4. **Performance Analysis & Profiling**
5. **Advanced Parallel Algorithms**
6. **Comparison of GPU Programming Frameworks** (PyTorch, Numba, Triton)

## Usage Examples

### Running Performance Comparisons
```bash
cd Codes/Lecture1/
python3 pytorch_square.py    # PyTorch implementations
python3 numba_square.py      # Numba CUDA kernels
python3 triton_square.py     # Triton implementations
```

### Exploring PMPP Labs
```bash
cd Books/PMPP/InstructorResources/Instructor\ Solution\ Manual,\ Labs\ with\ Solutions/Labs_with_Solutions/
# Each lab contains main.cu, kernel.cu, and support.cu files
```

## Learning Path

1. Start with `Codes/Lecture1/` for basic GPU programming concepts
2. Review `HandWrittenNotes/` for theoretical understanding
3. Practice with PMPP lab exercises in `Books/PMPP/`
4. Compare different implementation approaches using the provided examples
5. Use Docker environment for consistent development experience

## Contributing

This is a personal learning repository, but feel free to:
- Open issues for questions or discussions
- Suggest improvements to code examples
- Share additional resources or implementations

## Resources

- [CUDA MODE Course](https://github.com/cuda-mode)
- [Programming Massively Parallel Processors Book](https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0128119861)
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [PyTorch CUDA Documentation](https://pytorch.org/docs/stable/cuda.html)
