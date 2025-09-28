#include "__preprocessor__.h" // to use my core lib i had to -> mv .cpp .cu --- so when downloading lib first time you have to run ./start 2 time first with -c that solo
#include "openMP_test.h"


#ifdef BUILD_EXECUTABLE
int main(int argc, char* argv[])
{
    OpenMP_GPU_test();

    return 0;
}
#endif