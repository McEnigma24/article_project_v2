#pragma once

#include "__preprocessor__.h"
#include "parallel_common.h"
#include <span>
#include "consts.h"
#include "Sphere.hpp"

GPU_LINE(__host__ __device__)
void soft_max_value(Sphere* input, u64 size, const double param = 1)
{
    double exp_from_input[neighbor_count];         for(int i=0; i < neighbor_count; i++) exp_from_input[i] = 0;
    double sum = 0;

    for(int i=0; i < size; i++)
    {
        double e = exp( input[i].t / param ); // to może nie działać na GPU //
        
        exp_from_input[i] = e;
        sum += e;
    }

    // has to separate loop so we have a sum //
    
    for(int i=0; i < size; i++)
    {
        input[i].t = ( exp_from_input[i] / sum );
    }
}

GPU_LINE(__host__ __device__)
double random_0_1(unsigned long seed, int threadIdx)
{
    #if defined(__CUDA_ARCH__) && defined(GPU)
        curandState state;
        curand_init(seed, threadIdx, 0, &state);
    
        return curand_uniform(&state);
    #else
        return ( std::rand() / ((double) RAND_MAX) );
    #endif
}

GPU_LINE(__host__ __device__)
int pick_based_on_provided_chance(Sphere* element_list, u64 size, unsigned long seed, int threadIdx)
{
    for(;;) // we keep on trying until we pick something //
    {
        double random = random_0_1(seed, threadIdx);
        double running_sum = 0;
        for(int i=0; i < size; i++)
        {
            running_sum += element_list[i].t;
            if(random <= running_sum)
            {
                return element_list[i].id;
            }
        }

        // we come here if values do not sum up to 1 // - rest of the range is undefined, so we just pick
        // one more time, until we strike a chance defined value

        CPU_LINE(line("looking for another pick"));
    }

    return {};
}