#include "__preprocessor__.h" // to use my core lib i had to -> mv .cpp .cu --- so when downloading lib first time you have to run ./start 2 time first with -c that solo
#include "openMP_test.h"
#include "Multi_Dimension_View_Array.hpp"
#include "fstream"

#define sphere_radius ( 1.0f )

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

GPU_LINE(__host__)
GPU_LINE(__device__)
GPU_LINE(__device__ __host__)

typedef float unit;
// typedef double unit;

struct ovito_XYZ_format_obj
{
    int id;
    unit x, y, z;

    GPU_LINE(__device__ __host__)
    void init(int id, unit x, unit y, unit z)
    {
        this->id = id;
        this->x = x * 2.4f;
        this->y = y * 2.4f;
        this->z = z * 2.4f;
    }

    static void dumpToFile(const ovito_XYZ_format_obj* arr, size_t size, const string& FILEPATH = "output/test.xyz")
    {
        std::ofstream fout(FILEPATH, std::ios::out | std::ios::trunc);
        if (!fout)
        {
            std::cerr << "Nie można otworzyć pliku do nadpisania: " << FILEPATH << "\n";
            return;
        }

        fout << size << endl;
        fout << " " << endl;
        for(int i=0; i<size; i++)
        {
            const auto& obj = arr[i];
            fout << obj.x << " " << obj.y << " " << obj.z << " " << obj.id << endl;
        }

        if (!fout)
        {
            std::cerr << "Błąd podczas zapisu do pliku: " << FILEPATH << "\n";
            fout.close();
        }
        fout.close();
    }
};

class Sphere : public ovito_XYZ_format_obj
{
    
};

void ovito_xyz_creator(ovito_XYZ_format_obj& obj)
{
    var(obj.id);
    var(obj.x);
    var(obj.y);
    var(obj.z);
}

#ifdef BUILD_EXECUTABLE
int main(int argc, char* argv[])
{
    OpenMP_GPU_test();

    Multi_Dimension_View_Array<Sphere> arr;
    arr.set_sizes(3, 3, 3);

    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            for(int k=0; k<3; k++)
                arr.get(i, j, k)->init(rand() % 3, i, j, k);

    ovito_XYZ_format_obj::dumpToFile(arr.getBuffer().data(), arr.getBuffer().size());

    // zrobimy zwykły automat komórkowy montecarlo tylko 3D - w wrzucimy w Avitoo

    // potem możemy przemyśleć to przesunięcie i takie okrągłe sąsiedztwo (w sumie to będzie to samo, tylko przesunięte o ileś tam)


    return 0;
}
#endif