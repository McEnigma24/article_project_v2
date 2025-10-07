#include "__preprocessor__.h" // to use my core lib i had to -> mv .cpp .cu --- so when downloading lib first time you have to run ./start 2 time first with -c that solo
#include "openMP_test.h"
#include "Multi_Dimension_View_Array.hpp"
#include "fstream"
#include "RGB.hpp"
#include "min_max.hpp"

#define sphere_radius ( 1.0f )
#define ID_RANGE ( 25 )

// °C to K
#define ZERO_CELC_IN_KELV ( u(273.15) )

struct Sphere
{
    int id;
    unit t;

    GPU_LINE(__device__ __host__)
    void init(int id)
    {
        this->id = id;
        this->t = (std::rand() % 40) + ZERO_CELC_IN_KELV - 20; // 20 °C
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

void initialize_sim(Multi_Dimension_View_Array<Sphere>& arr)
{
    for(int z=0; z<arr.get_depth(); z++)
        for(int y=0; y<arr.get_height(); y++)
            for(int x=0; x<arr.get_width(); x++)
                arr.get(x, y, z)->init((rand() % ID_RANGE));
}

template<size_t N>
class ObjTracker
{
    Multi_Dimension_View_Array<Sphere> arr_of_objects[N];
    Sphere* all_chunks_ptr;
    size_t current_array_index;

    size_t next_value()
    {
        return (current_array_index + 1) % N;
    }

public:

    ObjTracker(i64 width, i64 height, i64 depth)
    {
        set(width, height, depth);
    }

    GPU_LINE(__host__ __device__)
    void set(i64 width, i64 height, i64 depth, Sphere* pre_allocation = nullptr)
    {
        u64 count_spheres_in_one_iteration = width * height * depth;
        u64 count_all_iterations = count_spheres_in_one_iteration * N;
        CPU_LINE(var(CORE::humanReadableBytes(count_all_iterations * sizeof(Sphere)));)

        current_array_index = 0;

        // prealokacja //

        if(pre_allocation) all_chunks_ptr = pre_allocation;
        else all_chunks_ptr = new Sphere[count_all_iterations];   // wali cały jeden duży placek

        Sphere* current_chunk = all_chunks_ptr;

        for(size_t i=0; i<N; i++)
        {
            arr_of_objects[i].init(width, height, depth, current_chunk); // tutaj dla CPU można podać Null i będzie przypadek dla obszarów w różnych miejscach pamięci
            current_chunk += count_spheres_in_one_iteration;

            initialize_sim(arr_of_objects[i]);
        }
    }

    size_t get_total_allocated_size()
    {
        return arr_of_objects[0].get_width() * arr_of_objects[0].get_height() * arr_of_objects[0].get_depth() * N;
    }

    Sphere* get_all_chunks_ptr()
    {
        return all_chunks_ptr;
    }

    ~ObjTracker()
    {
        CPU_LINE(if(all_chunks_ptr) delete[] all_chunks_ptr);
    }

    Multi_Dimension_View_Array<Sphere>& get_current_obj()
    {
        return arr_of_objects[current_array_index];
    }

    Multi_Dimension_View_Array<Sphere>& get_next_obj()
    {
        return arr_of_objects[next_value()];
    }

    void next_cycle()
    {
        current_array_index = next_value();
    }

    void reset_to_start()
    {
        current_array_index = 0;
    }
};

template<size_t N>
void per_sphere(ObjTracker<N>& obj_tracker, const coords& my_coords)
{
    auto& current_array = obj_tracker.get_current_obj();
    auto& next_arr = obj_tracker.get_next_obj();
    
    const auto& current_obj = *current_array.get(my_coords.x, my_coords.y, my_coords.z);
    auto& next_obj = *next_arr.get(my_coords.x, my_coords.y, my_coords.z);
    
    unit biggest_temp = current_obj.t;
    int biggest_temp_id = current_obj.id;
    
    constexpr int neighbor_range = 1;
    constexpr int neighbor_width = (neighbor_range * 2) + 1;
    constexpr int neighbor_count = neighbor_width * neighbor_width * neighbor_width - 1;
    // średnia z temperatury sąsiadów -> liczymy delte i można ją zparametryzować
    // żeby nie od razu jakby własna temperatura stawała się taka sama jak otoczenia

    Average<unit> neighbor_avg_temp;

    // pętla po sąsiadach - Moora 3D
    for(int dz=-neighbor_range; dz<=neighbor_range; dz++)
        for(int dy=-neighbor_range; dy<=neighbor_range; dy++)
            for(int dx=-neighbor_range; dx<=neighbor_range; dx++)
            {
                if(dx == 0 && dy == 0 && dz == 0) continue;

                int nx = ((my_coords.x + dx) + current_array.get_width())     % current_array.get_width();
                int ny = ((my_coords.y + dy) + current_array.get_height())    % current_array.get_height();
                int nz = ((my_coords.z + dz) + current_array.get_depth())     % current_array.get_depth();

                const Sphere* neighbor = current_array.get(nx, ny, nz);

                neighbor_avg_temp.add(neighbor->t);

                if(biggest_temp < neighbor->t)
                {
                    biggest_temp = neighbor->t;
                    biggest_temp_id = neighbor->id;
                }
            }

    // base line //
    next_obj.t = current_obj.t;
    
    unit full_delta = neighbor_avg_temp.get() - current_obj.t;
    next_obj.t += (full_delta * (0.75)); // 75% zmiany do osiągnięcia średniej sąsiadów

    // dodatek z zewnątrz //
    const u64 MAX = current_array.get_width() - 1;
    constexpr u64 range = u(10); // ogrzewanie na jaka głębokość w ilości sfer
    if ( value_between((u64)0, my_coords.x, range)
      || value_between((MAX - range), my_coords.x, MAX))
    {
        u64 distance_to_Heat_Source = (value_between((u64)0, my_coords.x, range))
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

template<size_t N>
void dump_all_saved_states_to_file(ObjTracker<N>& obj_tracker)
{
    obj_tracker.reset_to_start();

    for(size_t i=0; i<N; i++)
    {
        Sphere::dump_to_file(obj_tracker.get_current_obj());
        obj_tracker.next_cycle();
    }
}





constexpr int cube_side = 90;
constexpr int sim_steps = 100;

__global__ void init_ObjTracker(ObjTracker<sim_steps>* dev_objTracker, i64 width, i64 height, i64 depth, Sphere* dev_all_chunks)
{
    dev_objTracker->set(width, height, depth, dev_all_chunks);
}

#ifdef BUILD_EXECUTABLE
int main(int argc, char* argv[])
{
    srand(time(NULL));
    // ObjTracker - mogę mu podać tyle samo co sim_steps, wtedy zapisze jak już będzie po wszystkim -> albo batch, po którym zapiszę wszystko do pliku

    ObjTracker<sim_steps> obj_tracker(cube_side, cube_side, cube_side);

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
                    per_sphere(obj_tracker, arr.get_coords(i));
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

        size_t all_chunks_total_size_count = obj_tracker.get_total_allocated_size();

        // Alokacja - obj trackera //
        ObjTracker<sim_steps>* dev_obj_tracker = nullptr;
        CCE(cudaMalloc((void**)&dev_obj_tracker, sizeof(ObjTracker<sim_steps>)));
        CCE(cudaMemcpy(dev_obj_tracker, &obj_tracker, sizeof(ObjTracker<sim_steps>), cudaMemcpyHostToDevice));

        // Alokacja - pamięci pod spodem na iteracje //
        Sphere* dev_all_chunks = nullptr;
        CCE(cudaMalloc((void**)&dev_all_chunks, all_chunks_total_size_count));
        CCE(cudaMemcpy(dev_all_chunks, obj_tracker.get_all_chunks_ptr(), all_chunks_total_size_count * sizeof(Sphere), cudaMemcpyHostToDevice));

        auto width = obj_tracker.get_current_obj().get_width();
        auto height = obj_tracker.get_current_obj().get_height();
        auto depth = obj_tracker.get_current_obj().get_depth();

        init_ObjTracker<<<1, 1>>>(dev_obj_tracker, width, height, depth, dev_all_chunks);
        CCE(cudaDeviceSynchronize());

        // trzeba by zrobić tak, że Kernel dodaje już obiekt -> obj trackera -> którego pointery już pokazują na właściwe miejsca na dev //

        // albo puszczamy 1 kernel na start - on ustawia i reszta korzysta - potrzebujem tylko, żeby ustawić pointery w obj_trackerze na
        // on dev miejsca w pamięci

        // gpu_kernel<<<cube_side, cube_side, cube_side>>>(dev_obj_tracker);

        // wywołanie kernela - dev_obj_tracker, width, height, depth, dev_all_chunks

        // dev_obj_tracker->set(width, height, depth, dev_all_chunks)

        // a potem reszta już rusza


        CCE(cudaDeviceSynchronize());


        // kopiujemy z powrotem output //
        CCE(cudaMemcpy(obj_tracker.get_all_chunks_ptr(), dev_all_chunks, all_chunks_total_size_count * sizeof(Sphere), cudaMemcpyDeviceToHost));

        CCE(cudaFree(dev_all_chunks));
        CCE(cudaFree(dev_obj_tracker));

        time_stamp("GPU - preparation - FINISHED");
        time_stamp("GPU - cleanup - FINISHED");
        time_stamp("GPU - FINISH");
    }
    #endif


    // DO TEGO MOMENTU JUŻ DAJĘ ZNAĆ GRUBEMU //

    // potem możemy przemyśleć to przesunięcie i takie okrągłe sąsiedztwo (w sumie to będzie to samo, tylko przesunięte o ileś tam)
    // przesunięcie o pół sfery w jedną stronę, a cykliczność indexów zrobi swoje

    return 0;
}
#endif