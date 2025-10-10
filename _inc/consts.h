#pragma once

#define sphere_radius ( 1.0f )
#define ID_RANGE ( 5 )

constexpr int cube_side = 50;
constexpr int sim_steps = 100;

// °C to K
#define ZERO_CELC_IN_KELV ( u(273.15) )


// ogrzewanie na jaka głębokość w ilości sfer
#define heat_penetration_range ( 10u )

// 75% zmiany w stronę osiągnięcia średniej sąsiadów //
#define temp_adaptation_to_neighbors_factor ( 0.75 )
#define soft_max_param ( 0.1 )

#define constant_cooling ( 0.1 )
#define sides_heating ( 10 )

// / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / //

#define TEMP_CHANGES

// #define ID_CHANGES_MAX_TEMP
#define ID_CHANGES_SOFTMAXING_SUMMED_TEMP
