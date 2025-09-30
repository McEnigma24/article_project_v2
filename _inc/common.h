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