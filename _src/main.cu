#include "__preprocessor__.h" // to use my core lib i had to -> mv .cpp .cu --- so when downloading lib first time you have to run ./start 2 time first with -c that solo
#include "openMP_test.h"
#include "Multi_Dimension_View_Array.hpp"
#include "fstream"
#include "RGB.hpp"
#include "min_max.hpp"
#include "selection_func.hpp"
#include "consts.h"
#include "Sphere.hpp"


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
        var(CORE::humanReadableBytes(count_all_iterations * sizeof(Sphere)));

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

            if(i == 0)
                initialize_sim(arr_of_objects[i], 0);
        }

        // line("ObjTracker - init complete");
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

            if(initialize && i == 0)
                initialize_sim(arr_of_objects[i], seed);
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
        // CPU_LINE(if(all_chunks_ptr) delete[] all_chunks_ptr);
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

struct id_temp_Controller
{
    Sphere summed_temp_for_each_id[neighbor_count];

    GPU_LINE(__device__ __host__)
    void init()
    {
        for(int i=0; i<neighbor_count; i++)
        {
            summed_temp_for_each_id[i].id = 0;
            summed_temp_for_each_id[i].t = 0;
        }
    }

    GPU_LINE(__device__ __host__)
    void add(const Sphere& obj)
    {
        for(int i=0; i<neighbor_count; i++)
        {
            auto& current = summed_temp_for_each_id[i];

            if(current.id == obj.id)
            {
                current.t += obj.t;
                return;
            }
            else if(current.id == 0)
            {
                current.id = obj.id;
                current.t = obj.t;
                return;
            }

            // czyli zapycha od początku - jak natrafi na 0, to tam wstawi //
        }
    }

    GPU_LINE(__device__ __host__)
    unit add_up_all_temps()
    {
        unit total = 0;
        for(int i=0; i<neighbor_count; i++)
        {
            total += summed_temp_for_each_id[i].t;
        }
        return total;
    }

    GPU_LINE(__device__ __host__)
    unit biggest_summed_temp()
    {
        unit biggest_temp = 0;
        for(int i=0; i<neighbor_count; i++)
        {
            if(biggest_temp < summed_temp_for_each_id[i].t)
            {
                biggest_temp = summed_temp_for_each_id[i].t;
            }
        }
        return biggest_temp;
    }

    GPU_LINE(__device__ __host__)
    void devide_by(const unit& num)
    {
        for(int i=0; i<neighbor_count; i++)
        {
            summed_temp_for_each_id[i].t /= num;
        }
    }

    GPU_LINE(__device__ __host__)
    Sphere* get_tab()
    {
        return summed_temp_for_each_id;
    }

    GPU_LINE(__device__ __host__)
    u64 get_size()
    {
        for(int i=0; i<neighbor_count; i++)
        {
            if(summed_temp_for_each_id[i].id == 0)
                return i;
        }
        return neighbor_count;
    }
};

GPU_LINE(__device__ __host__)
void per_sphere(unsigned long seed, int step, Sphere* current_array, Sphere* next_array, int i, u64 width, u64 height, u64 depth)
{
    auto& current_obj = current_array[i];
    auto& next_obj = next_array[i];

    coords my_coords = get_coords(i, width, height);

    #ifdef ID_CHANGES_MAX_TEMP
        unit biggest_temp = current_obj.t;
        int biggest_temp_id = current_obj.id;
    #endif

    id_temp_Controller controller;
    controller.init();
    
    // pętla po sąsiadach - Moora 3D
    for(int dz=-neighbor_range; dz<=neighbor_range; dz++)
        for(int dy=-neighbor_range; dy<=neighbor_range; dy++)
            for(int dx=-neighbor_range; dx<=neighbor_range; dx++)
    {
        if(dx == 0 && dy == 0 && dz == 0) continue;

        int nx = ((my_coords.x + dx) + width)     % width;
        int ny = ((my_coords.y + dy) + height)    % height;
        int nz = ((my_coords.z + dz) + depth)     % depth;

        const Sphere& neighbor = current_array[d3tod1(nx, ny, nz, width, height)];

        #ifdef ID_CHANGES_MAX_TEMP
            if(biggest_temp < neighbor.t)
            {
                biggest_temp = neighbor.t;
                biggest_temp_id = neighbor.id;
            }
        #endif

        controller.add(neighbor);
    }



    #ifdef TEMP_CHANGES
        next_obj.t = current_obj.t; // base line //

        unit neighbor_temp_sum = controller.add_up_all_temps();
        unit neighbor_temp_avg = neighbor_temp_sum / u(neighbor_count);

        unit full_delta = neighbor_temp_avg - current_obj.t;
        next_obj.t += (full_delta * (temp_adaptation_to_neighbors_factor));

        // dodatek z zewnątrz //
        const u64 MAX = width - 1;
        if ( value_between(0, my_coords.x, heat_penetration_range)
            || value_between((MAX - heat_penetration_range), my_coords.x, MAX))
        {
            u64 distance_to_Heat_Source = (value_between(0, my_coords.x, heat_penetration_range))
                                        ? my_coords.x
                                        : (MAX - my_coords.x);
            unit temp_increase = (u(heat_penetration_range - distance_to_Heat_Source) / u(heat_penetration_range)) * u(sides_heating);
            next_obj.t += temp_increase;
        }

        next_obj.t -= u(constant_cooling); // potem można bardziej zaawansowany //
                                           // np. sprawdzanie jak daleko masz do krawędzi ze współrzędnych //

        next_obj.t = (next_obj.t > 1.0) ? next_obj.t : 1.0; // temp always bigger then absolute zero // not exacly zero to 0.1 to avoid division by zero //
    #endif

    #ifdef ID_CHANGES_MAX_TEMP
        next_obj.id = biggest_temp_id;
    #endif

    #ifdef ID_CHANGES_SOFTMAXING_SUMMED_TEMP

        // teraz jeszcze możemy dodać liniową zależność, że tym większa szansa na zmianę im wyższa temperatura obiektu //
        // im wyższa temperatura, tym bardziej "aktywny" jest obiekt i chętniej zmienia ID //

        unit temp_normalized = current_obj.t / temp_threshold_for_id_change; // normalizacja względem progu
        unit change_probability = temp_normalized / (temp_normalized + u(1.0)); // sigmoid-like [0, 1]

        #if defined(__CUDA_ARCH__) && defined(GPU)
            curandState state;
            curand_init(seed + i, i, 0, &state);
            unit random_chance = curand_uniform(&state);
        #else
            unit random_chance = std::rand() / u(RAND_MAX);
        #endif
        
        if(!(random_chance < change_probability))
        {
            next_obj.id = current_obj.id; // zachowaj obecne ID
        }
        else
        {
            controller.add(current_obj);
            // summed_temp_for_each_id[current_obj.id] += current_obj.t; // taking into consideration self temperature //

            // summed_temp_for_each_id to rozpiska id -> i suma temperatur obiektów z każdej grupy id //
            // [id] -> temp_sum

            // teraz mamy absolutną sumę temperatur dla każdego id //
            // jeśli damy średnią to przestaniemy uwzględniać ile kuli tego samego id jest na około //

            unit biggest_summed_temp = controller.biggest_summed_temp();
            
            // scaling to [0, 1] so soft max does not explode from exp(x) //
            controller.devide_by(biggest_summed_temp);

            soft_max_value(controller.get_tab(), controller.get_size(), soft_max_param);
            // CPU_LINE(var(controller.get_size());)
            // CPU_LINE(var(neighbor_count));

            // teraz temperatury zmieniły się w prawdopodobieństwo //

            int picked_id = pick_based_on_provided_chance(controller.get_tab(), controller.get_size(), seed, i);

            CPU_LINE(line("\n\n"));
            for(int i=0; i<controller.get_size(); i++)
            {
                CPU_LINE(line("id: " + std::to_string(controller.get_tab()[i].id) + " chance: " + std::to_string(controller.get_tab()[i].t));)
            }

            next_obj.id = picked_id;
        }

    #endif
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

#ifdef GPU
    __global__ void kernel_Calculations(Sphere** dev_tab_of_chunks, u64 width, u64 height, u64 depth, unsigned long seed, int step)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;

        u64 total_spheres = width * height * depth;
        if (i >= total_spheres) return;

        per_sphere(seed, step, dev_tab_of_chunks[step], dev_tab_of_chunks[step + 1], i, width, height, depth);
    }
#endif

#ifdef BUILD_EXECUTABLE
int main(int argc, char* argv[])
{
    srand(time(NULL));
    // ObjTracker - mogę mu podać tyle samo co sim_steps, wtedy zapisze jak już będzie po wszystkim -> albo batch, po którym zapiszę wszystko do pliku

    ObjTracker obj_tracker(sim_width, sim_height, sim_depth);

    time_stamp_reset();

    #ifdef CPU
    {
        #pragma omp parallel
        {
            omp_set_num_threads(1);

            srand(time(NULL));
            for(int step = 0; step < (sim_steps - 1); step++)
            {
                auto& arr = obj_tracker.get_current_obj();
                
                #pragma omp for schedule(static) 
                for(int i=0; i<arr.get_total_number(); i++)
                {
                    per_sphere(0, step, obj_tracker.get_current_obj().get(0), obj_tracker.get_next_obj().get(0), i,
                                arr.get_width(), arr.get_height(), arr.get_depth()
                    );
                }

                #pragma omp barrier
                #pragma omp single
                {
                    line("CPU - next cycle " + std::to_string(step));
                    obj_tracker.next_cycle();
                }
            }
        }
        time_stamp("CPU - FINISH");
    
        dump_all_saved_states_to_file(obj_tracker);

        time_stamp("CPU - io DONE");
    }
    #endif

    // #ifdef GPU
    // {
    //     CCE(cudaSetDevice(0));

    //     size_t bytesize_of_one_iteration = obj_tracker.get_size_of_one_iteration() * sizeof(Sphere);

    //     // Alokacja - wypełniamy tablicę po stronie HOST, pointerami do chunków po stronie DEVICE //
    //     std::array<Sphere*, sim_steps> dev_arr_of_chunks;
    //     for(auto& dev_chunk_ptr : dev_arr_of_chunks)
    //     {
    //         CCE(cudaMalloc((void**)&dev_chunk_ptr, bytesize_of_one_iteration));
            
    //         static bool first = true;
    //         if(first)
    //         {
    //             first = false;
    //             CCE(cudaMemcpy(dev_chunk_ptr, obj_tracker.get_current_obj().get_data(), bytesize_of_one_iteration, cudaMemcpyHostToDevice));
    //         }
    //     }

    //     // Alokacja - tablicy chunków na DEVICE i skopiowanie pointerów jakie dostaliśmy //
    //     Sphere** dev_tab_of_chunks = nullptr;
    //     CCE(cudaMalloc((void**)&dev_tab_of_chunks, sim_steps * sizeof(Sphere*)));
    //     CCE(cudaMemcpy(dev_tab_of_chunks, dev_arr_of_chunks.data(), sim_steps * sizeof(Sphere*), cudaMemcpyHostToDevice));

    //     auto width = obj_tracker.get_current_obj().get_width();
    //     auto height = obj_tracker.get_current_obj().get_height();
    //     auto depth = obj_tracker.get_current_obj().get_depth();

    //     time_stamp("GPU - allocations / memcopies");

    //     int BLOCK_SIZE = 128;
	//     int NUMBER_OF_BLOCKS = obj_tracker.get_size_of_one_iteration() / BLOCK_SIZE + 1;

    //     for(int step = 0; step < (sim_steps - 1); step++)
    //     {
    //         // var(step);
    //         kernel_Calculations<<<NUMBER_OF_BLOCKS, BLOCK_SIZE>>>(dev_tab_of_chunks, width, height, depth, 1, step);
    //         CCE(cudaDeviceSynchronize()); // Synchronizacja między krokami

    //         line("GPU - next cycle " + std::to_string(step));
    //     }
    //     time_stamp("GPU - FINISH");


    //     // kopiujemy z powrotem outputem //
    //     std::array<Sphere*, sim_steps> host_tab_of_chunks;
    //     for(int i=0; i<dev_arr_of_chunks.size(); i++)
    //     {
    //         auto& dev_chunk_ptr = dev_arr_of_chunks[i];

    //         Sphere* host_one_chunk = new Sphere[bytesize_of_one_iteration];
    //         CCE(cudaMemcpy(host_one_chunk, dev_chunk_ptr, bytesize_of_one_iteration, cudaMemcpyDeviceToHost));

    //         host_tab_of_chunks[i] = host_one_chunk; // teraz std::array ma pointery do HOST side chunków

    //         CCE(cudaFree(dev_chunk_ptr));
    //     }
    //     time_stamp("GPU - copying outpus back to HOST");

    //     CCE(cudaFree(dev_tab_of_chunks));

    //     time_stamp("GPU - cleanup");



    //     obj_tracker.set_with_preallocated_tab(width, height, depth, host_tab_of_chunks.data(), 0, false);
    //     dump_all_saved_states_to_file(obj_tracker);

    //     time_stamp("GPU - io DONE");
    // }
    // #endif

    return 0;
}
#endif