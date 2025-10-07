#include "__preprocessor__.h" // to use my core lib i had to -> mv .cpp .cu --- so when downloading lib first time you have to run ./start 2 time first with -c that solo
#include "openMP_test.h"
#include "Multi_Dimension_View_Array.hpp"
#include "fstream"
#include "RGB.hpp"
#include "min_max.hpp"

#define sphere_radius ( 1.0f )
#define ID_RANGE ( 5 )

constexpr int cube_side = 90;
constexpr int sim_steps = 10;

// °C to K
#define ZERO_CELC_IN_KELV ( u(273.15) )

struct Sphere
{
    int id;
    unit t;
    
    GPU_LINE(__device__ __host__)
    void init(unsigned long seed)
    {
        #ifdef __CUDA_ARCH__
            curandState state;
            curand_init(seed, threadIdx.x, 0, &state);

            this->id = ((int)(curand_uniform(&state) * ID_RANGE)) % ID_RANGE;
            this->t = (int)(curand_uniform(&state) * 40) + ZERO_CELC_IN_KELV - 20;
        #else
            this->id = rand() % ID_RANGE;
            this->t = (std::rand() % 40) + ZERO_CELC_IN_KELV - 20; // 20 °C
        #endif
    }

    // x, y, z - id - t //
    static void dump_to_file(Multi_Dimension_View_Array<Sphere>& arr)
    {
        CPU_LINE(const string& FILEPATH = "output/cpu.xyz";)
        GPU_LINE(const string& FILEPATH = "output/gpu.xyz";)

        std::ofstream fout(FILEPATH, std::ios::out | std::ios::app);
        if (!fout)
        {
            std::cerr << "Nie można otworzyć pliku do nadpisania: " << FILEPATH << "\n";
            return;
        }

        fout << arr.get_total_number() << endl;
        fout << " " << endl;
        for(int z=0; z<arr.get_depth(); z++)
            for(int y=0; y<arr.get_height(); y++)
                for(int x=0; x<arr.get_width(); x++)
        {
            const auto& obj = *arr.get(x, y, z);

            char buf[256];
            int len = std::snprintf(//  x  y  z id  t
                    buf, sizeof(buf), "%d %d %d %d %f\n",
                    x * 2, y * 2, z * 2, obj.id, obj.t
            );
            if (len > 0)
            {
                fout << buf;
            }
        }

        if (!fout)
        {
            std::cerr << "Błąd podczas zapisu do pliku: " << FILEPATH << "\n";
            fout.close();
        }
        fout.close();
    }
};

GPU_LINE(__host__ __device__)
void initialize_sim(Multi_Dimension_View_Array<Sphere>& arr, unsigned long seed)
{
    for(int z=0; z<arr.get_depth(); z++)
        for(int y=0; y<arr.get_height(); y++)
            for(int x=0; x<arr.get_width(); x++)
                arr.get(x, y, z)->init(seed);
}

GPU_LINE(__host__ __device__)
bool value_between(const u64& lv, const u64& v, const u64& rv)
{
    return (lv <= v) && (v <= rv);
}

class ObjTracker
{
    Multi_Dimension_View_Array<Sphere> arr_of_objects[sim_steps];
    Sphere* all_chunks_ptr;
    size_t current_array_index;

    size_t next_value()
    {
        return (current_array_index + 1) % sim_steps;
    }

public:

    ObjTracker(i64 width, i64 height, i64 depth)
    {
        set(width, height, depth);
    }

    void set(i64 width, i64 height, i64 depth, Sphere* pre_allocation = nullptr)
    {
        u64 count_spheres_in_one_iteration = width * height * depth;
        u64 count_all_iterations = count_spheres_in_one_iteration * sim_steps;
        CPU_LINE(var(CORE::humanReadableBytes(count_all_iterations * sizeof(Sphere)));)

        current_array_index = 0;

        // prealokacja //

        // if(pre_allocation) all_chunks_ptr = pre_allocation;
        // else all_chunks_ptr = new Sphere[count_all_iterations];

        // Sphere* current_chunk = all_chunks_ptr;

        for(size_t i=0; i<sim_steps; i++)
        {
            // arr_of_objects[i].init(width, height, depth, current_chunk); // tutaj dla CPU można podać Null i będzie przypadek dla obszarów w różnych miejscach pamięci
            arr_of_objects[i].init(width, height, depth, nullptr); // teraz kolejne kroki iteracji są w różnych miejscach
            // current_chunk += count_spheres_in_one_iteration;

            CPU_LINE(initialize_sim(arr_of_objects[i]);)
        }
    }

    GPU_LINE(__host__ __device__)
    void set_with_preallocated_tab(i64 width,i64 height, i64 depth, Sphere** tab_of_preallocations, unsigned long seed, bool initialize = true)
    {
        u64 count_spheres_in_one_iteration = width * height * depth;
        u64 count_all_iterations = count_spheres_in_one_iteration * sim_steps;
        CPU_LINE(var(CORE::humanReadableBytes(count_all_iterations * sizeof(Sphere)));)

        current_array_index = 0;

        // prealokacja //
        for(size_t i=0; i<sim_steps; i++)
        {
            arr_of_objects[i].init(width, height, depth, tab_of_preallocations[i]);

            if(initialize && i == 0) initialize_sim(arr_of_objects[i], seed);
        }
    }

    size_t get_size_of_one_iteration()
    {
        return arr_of_objects[0].get_width() * arr_of_objects[0].get_height() * arr_of_objects[0].get_depth();
    }

    size_t get_size_of_all_iterations()
    {
        return get_size_of_one_iteration() * sim_steps;
    }

    Sphere* get_all_chunks_ptr()
    {
        return all_chunks_ptr;
    }

    ~ObjTracker()
    {
        CPU_LINE(if(all_chunks_ptr) delete[] all_chunks_ptr);
    }

    GPU_LINE(__host__ __device__)
    Multi_Dimension_View_Array<Sphere>& get_current_obj()
    {
        return arr_of_objects[current_array_index];
    }

    GPU_LINE(__host__ __device__)
    Multi_Dimension_View_Array<Sphere>& get_next_obj()
    {
        return arr_of_objects[next_value()];
    }

    GPU_LINE(__host__ __device__)
    Multi_Dimension_View_Array<Sphere>& get_obj(int i)
    {
        return arr_of_objects[i];
    }

    GPU_LINE(__host__ __device__)
    void next_cycle()
    {
        current_array_index = next_value();
    }

    GPU_LINE(__host__ __device__)
    void reset_to_start()
    {
        current_array_index = 0;
    }
};

GPU_LINE(__host__ __device__)
u64 d3tod1(u64 x, u64 y, u64 z, u64 WIDTH, u64 HEIGHT)
{
    return x + y * (WIDTH) + z * (WIDTH * HEIGHT);
}

GPU_LINE(__host__ __device__)
coords get_coords(u64 index_1d, u64 WIDTH, u64 HEIGHT)
{
    coords c;
    c.z = index_1d / (WIDTH * HEIGHT);
    c.y = (index_1d - c.z * (WIDTH * HEIGHT)) / WIDTH;
    c.x = index_1d - c.z * (WIDTH * HEIGHT) - c.y * WIDTH;

    return c;
}

GPU_LINE(__host__ __device__)
// void per_sphere(ObjTracker& obj_tracker, const coords& my_coords)
void per_sphere(Sphere* current_array, Sphere* next_array, int i, u64 width, u64 height, u64 depth)
{
    // auto& current_array = obj_tracker.get_current_obj();
    // auto& next_arr = obj_tracker.get_next_obj();
    
    // const auto& current_obj = *current_array.get(my_coords.x, my_coords.y, my_coords.z);
    // auto& next_obj = *next_arr.get(my_coords.x, my_coords.y, my_coords.z);

    const auto& current_obj = current_array[i];
    auto& next_obj = next_array[i];

    coords my_coords = get_coords(i, width, height);

    
    unit biggest_temp = current_obj.t;
    int biggest_temp_id = current_obj.id;
    
    constexpr int neighbor_range = 1;
    constexpr int neighbor_width = (neighbor_range * 2) + 1;
    constexpr int neighbor_count = neighbor_width * neighbor_width * neighbor_width - 1;
    // średnia z temperatury sąsiadów -> liczymy delte i można ją zparametryzować
    // żeby nie od razu jakby własna temperatura stawała się taka sama jak otoczenia

    unit neighbor_avg_temp_sum = 0;
    size_t neighbor_avg_temp_count = 0;
    
    // pętla po sąsiadach - Moora 3D
    for(int dz=-neighbor_range; dz<=neighbor_range; dz++)
        for(int dy=-neighbor_range; dy<=neighbor_range; dy++)
            for(int dx=-neighbor_range; dx<=neighbor_range; dx++)
    {
        if(dx == 0 && dy == 0 && dz == 0) continue;

        int nx = ((my_coords.x + dx) + width)     % width;
        int ny = ((my_coords.y + dy) + height)    % height;
        int nz = ((my_coords.z + dz) + depth)     % depth;

        // const Sphere* neighbor = current_array.get(nx, ny, nz);
        const Sphere& neighbor = current_array[d3tod1(nx, ny, nz, width, height)];

        // neighbor_avg_temp.add(neighbor.t);

        neighbor_avg_temp_sum += neighbor.t;
        neighbor_avg_temp_count++;

        if(biggest_temp < neighbor.t)
        {
            biggest_temp = neighbor.t;
            biggest_temp_id = neighbor.id;
        }
    }

    // base line //
    next_obj.t = current_obj.t;

    // unit full_delta = neighbor_avg_temp.get() - current_obj.t;
    unit full_delta = (neighbor_avg_temp_sum / u(neighbor_avg_temp_count)) - current_obj.t;
    next_obj.t += (full_delta * (0.75)); // 75% zmiany do osiągnięcia średniej sąsiadów

    // dodatek z zewnątrz //
    const u64 MAX = width - 1;
    constexpr u64 range = 10; // ogrzewanie na jaka głębokość w ilości sfer

    if ( value_between(0, my_coords.x, range)
        || value_between((MAX - range), my_coords.x, MAX))
    {
        u64 distance_to_Heat_Source = (value_between(0, my_coords.x, range))
                                      ? my_coords.x
                                      : (MAX - my_coords.x);
        unit temp_increase = (u(range - distance_to_Heat_Source) / u(range)) * u(10); // max 10 K
        next_obj.t += temp_increase;
    }

    next_obj.id = biggest_temp_id;
    // tutaj można by zrobić rozkład - tak żeby uwzględnić ile było takich id w otoczeniu
    // i temperatury to jakby wagi, najliczniejszy z najwyższymi temperaturami ma największą szansą na przekonianie
    // aktualniej sfery żeby ona wzięła jego id
    // (też uwzględniamy to, że sfera może zostać przy swoim ID)

    // tutaj też trzeba dodać, że po prostu jeśli jest wyższa wartość absolutna
    // (nie tylko relatywna - w porównaniu ze średnią)
    // to ten ma większą szansę "przekonywania", tak żeby było widać, że w miejscach gorących robią się większ ziarna
}

void dump_all_saved_states_to_file(ObjTracker& obj_tracker)
{
    obj_tracker.reset_to_start();

    for(size_t i=0; i<sim_steps; i++)
    {
        Sphere::dump_to_file(obj_tracker.get_current_obj());
        obj_tracker.next_cycle();
    }
}

__global__ void init_ObjTracker(ObjTracker* dev_objTracker, i64 width, i64 height, i64 depth, Sphere** dev_tab_of_chunks, unsigned long seed)
{
    dev_objTracker->set_with_preallocated_tab(width, height, depth, dev_tab_of_chunks, seed);
}

__global__ void kernel_Calculations(Sphere** dev_tab_of_chunks, u64 width, u64 height, u64 depth)
{
    // __shared__ cuda::barrier<cuda::thread_scope_block> bar;

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // if (0 == i) {
    //     bar = cuda::barrier<cuda::thread_scope_block>(blockDim.x);
    // }
    // b.arrive_and_wait();

    for(int step = 0; step < (sim_steps - 1); step++)
    {
        per_sphere(dev_tab_of_chunks[step], dev_tab_of_chunks[step + 1], i, width, height, depth);
        // b.arrive_and_wait();
    }
}

#ifdef BUILD_EXECUTABLE
int main(int argc, char* argv[])
{
    srand(time(NULL));
    // ObjTracker - mogę mu podać tyle samo co sim_steps, wtedy zapisze jak już będzie po wszystkim -> albo batch, po którym zapiszę wszystko do pliku

    ObjTracker obj_tracker(cube_side, cube_side, cube_side);

    time_stamp_reset();

    #ifdef CPU
    {
        #pragma omp parallel
        {
            srand(time(NULL));
            for(int step = 0; step < (sim_steps - 1); step++)
            {
                auto& arr = obj_tracker.get_current_obj();
                
                #pragma omp for schedule(static)
                for(int i=0; i<arr.get_total_number(); i++)
                {
                    per_sphere(obj_tracker, i);
                }

                #pragma omp barrier
                #pragma omp single
                {
                    obj_tracker.next_cycle();
                }
            }
        }
        time_stamp("CPU - FINISH");
    
        dump_all_saved_states_to_file(obj_tracker);

        time_stamp("CPU - io DONE");
    }
    #endif

    #ifdef GPU
    {
        CCE(cudaSetDevice(0));

        size_t bytesize_of_one_iteration = obj_tracker.get_size_of_one_iteration() * sizeof(Sphere);

        // Alokacja - obj trackera //
        ObjTracker* dev_obj_tracker = nullptr;
        CCE(cudaMalloc((void**)&dev_obj_tracker, sizeof(ObjTracker)));
        CCE(cudaMemcpy(dev_obj_tracker, &obj_tracker, sizeof(ObjTracker), cudaMemcpyHostToDevice));

        // Alokacja - wypełniamy tablicę po stronie HOST, pointerami do chunków po stronie DEVICE //
        std::array<Sphere*, sim_steps> dev_arr_of_chunks;
        for(auto& dev_chunk_ptr : dev_arr_of_chunks)
        {
            static bool first = true;
            CCE(cudaMalloc((void**)&dev_chunk_ptr, bytesize_of_one_iteration));
        }

        // Alokacja - tablicy chunków na DEVICE i skopiowanie pointerów jakie dostaliśmy //
        Sphere** dev_tab_of_chunks = nullptr;
        CCE(cudaMalloc((void**)&dev_tab_of_chunks, sim_steps * sizeof(Sphere*)));
        CCE(cudaMemcpy(dev_tab_of_chunks, dev_arr_of_chunks.data(), sim_steps * sizeof(Sphere*), cudaMemcpyHostToDevice));

        auto width = obj_tracker.get_current_obj().get_width();
        auto height = obj_tracker.get_current_obj().get_height();
        auto depth = obj_tracker.get_current_obj().get_depth();

        time_stamp("GPU - allocations / memcopies");
        init_ObjTracker<<<1, 1>>>(dev_obj_tracker, width, height, depth, dev_tab_of_chunks, time(0));
        CCE(cudaDeviceSynchronize());
        time_stamp("GPU - prep");

        int BLOCK_SIZE = 128;
		int NUMBER_OF_BLOCKS = obj_tracker.get_size_of_one_iteration() / BLOCK_SIZE + 1;

        kernel_Calculations<<<NUMBER_OF_BLOCKS, BLOCK_SIZE>>>(dev_tab_of_chunks, width, height, depth);
        CCE(cudaDeviceSynchronize());
        time_stamp("GPU - FINISH");


        // kopiujemy z powrotem outputem //
        std::array<Sphere*, sim_steps> host_tab_of_chunks;
        for(int i=0; i<dev_arr_of_chunks.size(); i++)
        {
            auto& dev_chunk_ptr = dev_arr_of_chunks[i];

            Sphere* host_one_chunk = new Sphere[bytesize_of_one_iteration];
            CCE(cudaMemcpy(host_one_chunk, dev_chunk_ptr, bytesize_of_one_iteration, cudaMemcpyDeviceToHost));

            host_tab_of_chunks[i] = host_one_chunk; // teraz std::array ma pointery do HOST side chunków

            CCE(cudaFree(dev_chunk_ptr));
        }
        time_stamp("GPU - copying outpus back to HOST");

        CCE(cudaFree(dev_tab_of_chunks));
        CCE(cudaFree(dev_obj_tracker));

        time_stamp("GPU - cleanup");



        obj_tracker.set_with_preallocated_tab(width, height, depth, host_tab_of_chunks.data(), false);
        dump_all_saved_states_to_file(obj_tracker);

        time_stamp("GPU - io DONE");
    }
    #endif


    // do tego momentu wszystko z GPU działa //

    // potem możemy przemyśleć to przesunięcie i takie okrągłe sąsiedztwo (w sumie to będzie to samo, tylko przesunięte o ileś tam)
    // przesunięcie o pół sfery w jedną stronę, a cykliczność indexów zrobi swoje

    return 0;
}
#endif