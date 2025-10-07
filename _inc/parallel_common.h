#pragma once

#ifdef GPU
#define GPU_LINE(...) __VA_ARGS__
#else
#define GPU_LINE(...) 
#endif

#ifdef CPU
#define CPU_LINE(...) __VA_ARGS__
#else
#define CPU_LINE(...) 
#endif

#ifdef CPU
#include <omp.h>
#endif

#ifdef GPU
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"
// #include <cuda/barrier>
#endif


// typedef float unit;
typedef double unit;

#define u(x) ( static_cast<unit>(x) )