#define CCE(x)                                                                                                                                   \
{                                                                                                                                                \
    cudaError_t err = x;                                                                                                                         \
    if (err != cudaSuccess)                                                                                                                      \
    {                                                                                                                                            \
        const string error = "CUDA ERROR - " + std::to_string(__LINE__) + " : " + __FILE__ + "\n";                                               \
        cout << error;                                                                                                                           \
        exit(EXIT_FAILURE);                                                                                                                      \
    }                                                                                                                                            \
}

__global__ void test(int* a, int* b, int* result, int ARRAY_SIZE)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (!(i < ARRAY_SIZE)) return;

    result[i] = a[i] + b[i];
}

void OpenMP_GPU_test()
{
    int size = 100'000'000;
    int* a = new int[size];
    int* b = new int[size];
    int* result = new int[size];

    for (int i = 0; i < size; i++)
    {
        a[i] = i;
        b[i] = size - i;
    }

    time_stamp_reset();

    for (int i = 0; i < size; i++)
    {
        result[i] = a[i] + b[i];
    }
    time_stamp("Iterative");

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++)
    {
        result[i] = a[i] + b[i];
    }
    time_stamp("CPU Parallel");

    int byte_size = size * sizeof(int);
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_result = 0;
    CCE(cudaSetDevice(0));
    CCE(cudaMalloc((void**)&dev_a, byte_size));
    CCE(cudaMalloc((void**)&dev_b, byte_size));
    CCE(cudaMalloc((void**)&dev_result, byte_size));

    CCE(cudaMemcpy(dev_a, a, byte_size, cudaMemcpyHostToDevice));
    CCE(cudaMemcpy(dev_b, b, byte_size, cudaMemcpyHostToDevice));

    int BLOCK_SIZE = 64;
    int NUMBER_OF_BLOCKS = size / BLOCK_SIZE + 1;

    time_stamp_reset();
    test<<<NUMBER_OF_BLOCKS, BLOCK_SIZE>>>(dev_a, dev_b, dev_result, size);
    CCE(cudaDeviceSynchronize());
    time_stamp("GPU");

    CCE(cudaMemcpy(result, dev_result, byte_size, cudaMemcpyDeviceToHost));
    CCE(cudaFree(dev_a));
    CCE(cudaFree(dev_b));
    CCE(cudaFree(dev_result));
}