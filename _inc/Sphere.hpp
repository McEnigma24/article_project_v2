#pragma once

#include "__preprocessor__.h"
#include "consts.h"

struct Sphere
{
    int id;
    unit t;
    
    void init(unsigned long seed)
    {
        this->t = (std::rand() % 40) + ZERO_CELC_IN_KELV - 20; // 20 °C

        // static u64 id_counter = 1;

        this->id = (std::rand() % ID_RANGE) + 1;
        // this->id = id_counter++;
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