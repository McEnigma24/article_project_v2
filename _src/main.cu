#include "__preprocessor__.h" // to use my core lib i had to -> mv .cpp .cu --- so when downloading lib first time you have to run ./start 2 time first with -c that solo
#include "openMP_test.h"
#include "Multi_Dimension_View_Array.hpp"
#include "fstream"
#include "RGB.hpp"
#include "min_max.hpp"

#define sphere_radius ( 1.0f )
#define ID_RANGE ( 5 )

// °C to K
#define ZERO_CELC_IN_KELV ( u(273.15) )

struct Sphere
{
    int id;
    unit x, y, z;
    unit t;

    GPU_LINE(__device__ __host__)
    void init(int id, unit x, unit y, unit z)
    {
        this->id = id;
        this->x = x * 2.4f;
        this->y = y * 2.4f;
        this->z = z * 2.4f;
        this->t = (std::rand() % 40) + ZERO_CELC_IN_KELV - 20; // 20 °C
    }

    // x, y, z - id - t //
    static void dump_to_file(const Sphere* arr, size_t size)
    {
        CPU_LINE(const string& FILEPATH = "output/cpu.xyz";)
        GPU_LINE(const string& FILEPATH = "output/gpu.xyz";)

        std::ofstream fout(FILEPATH, std::ios::out | std::ios::app);
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
            char buf[256];
            int len = std::snprintf(//  x  y  z id  t
                    buf, sizeof(buf), "%lf %lf %lf %d %f\n",
                    obj.x, obj.y, obj.z, obj.id, obj.t
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

    static void dump_to_file(const Multi_Dimension_View_Array<Sphere>& arr)
    {
        dump_to_file(arr.get_vector().data(), arr.get_vector().size());
    }
};

void initialize_sim(Multi_Dimension_View_Array<Sphere>& arr)
{
    for(int z=0; z<arr.get_depth(); z++)
        for(int y=0; y<arr.get_height(); y++)
            for(int x=0; x<arr.get_width(); x++)
                arr.get(x, y, z)->init((rand() % ID_RANGE), x, y, z);
}

template<typename T, size_t N>
class ObjTracker
{
    T arr_of_objects[N];
    size_t current_array_index;

    size_t next_value()
    {
        return (current_array_index + 1) % N;
    }

public:

    ObjTracker(i64 width, i64 height, i64 depth)
        : current_array_index(0)
    {
        for(size_t i=0; i<N; i++)
        {
            arr_of_objects[i].set_sizes(width, height, depth);
            initialize_sim(arr_of_objects[i]);
        }
    }

    T& get_current_obj()
    {
        return arr_of_objects[current_array_index];
    }

    T& get_next_obj()
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
void per_sphere(ObjTracker<Multi_Dimension_View_Array<Sphere>, N>& obj_tracker, const coords& my_coords)
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
    
    unit full_delta = std::abs(neighbor_avg_temp.get() - current_obj.t);

    

    // tutaj jakąś deltę obliczyć z sądiadów i zastosować do siebie //
    next_obj.t = current_obj.t;



    // dodatek z zewnątrz //
    const unit MAX = current_array.get(current_array.get_width() - 1, my_coords.y, my_coords.z)->x;
    constexpr unit range = u(10); // ogrzewanie na jaka głębokość w ilości sfer
    if ( value_between((unit)0, current_obj.x, range)
      || value_between((unit)(MAX - range), (unit)current_obj.x, MAX))
    {
        unit distance_to_Heat_Source = (value_between((unit)0, current_obj.x, range)) 
                                          ? current_obj.x
                                          : (MAX - current_obj.x);
        unit temp_increase = (u(range - distance_to_Heat_Source) / u(range)) * u(10); // max 10 K
        next_obj.t += temp_increase;
    }
    next_obj.id = biggest_temp_id;
}

template<size_t N>
void dump_all_saved_states_to_file(ObjTracker<Multi_Dimension_View_Array<Sphere>, N>& obj_tracker)
{
    obj_tracker.reset_to_start();

    for(size_t i=0; i<N; i++)
    {
        Sphere::dump_to_file(obj_tracker.get_current_obj());
        obj_tracker.next_cycle();
    }
}

#ifdef BUILD_EXECUTABLE
int main(int argc, char* argv[])
{
    srand(time(NULL));
    constexpr int cube_side = 90;
    constexpr int sim_steps = 10;

    // ObjTracker - mogę mu podać tyle samo co sim_steps, wtedy zapisze jak już będzie po wszystkim -> albo batch, po którym zapiszę wszystko do pliku

    ObjTracker<Multi_Dimension_View_Array<Sphere>, sim_steps> obj_tracker(cube_side, cube_side, cube_side);

    time_stamp_reset();
    #pragma omp parallel
    {
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
    time_stamp("FINISH");
    
    dump_all_saved_states_to_file(obj_tracker);

    time_stamp("io DONE");

    // teraz któryś z tych początkowych stanów można zapisać albo zawsze do niego kilkoma iteracjami dochodzić
    // na takim stanie odpalić nagrzewanie z dwóch stron tych sfer
    // i przechodzić po sąsiadach, tylko tym razem sprawdzać temperaturę sąsiadów, i dołączać się do większej - tutaj już zobaczyć w Notatki co Gruby chciał




    // DO TEGO MOMENTU JUŻ DAJĘ ZNAĆ GRUBEMU //

    // potem możemy przemyśleć to przesunięcie i takie okrągłe sąsiedztwo (w sumie to będzie to samo, tylko przesunięte o ileś tam)
    // przesunięcie o pół sfery w jedną stronę, a cykliczność indexów zrobi swoje

    return 0;
}
#endif