#pragma once

#define sphere_radius ( 1.0f )
#define ID_RANGE ( 15 )

// constexpr int multiplier = 1;
// constexpr int multiplier = 10;
constexpr int multiplier = 3;

constexpr int sim_width = 25 * multiplier;
constexpr int sim_depth = 10 * multiplier;
constexpr int sim_height = 5 * multiplier;

constexpr int sim_steps = 300;
// constexpr int sim_steps = 100;

constexpr int neighbor_range = 1;
constexpr int neighbor_width = (neighbor_range * 2) + 1;
constexpr int neighbor_count = neighbor_width * neighbor_width * neighbor_width - 1;

// °C to K
#define ZERO_CELC_IN_KELV ( u(273.15) )


// ogrzewanie na jaka głębokość w ilości sfer
#define heat_penetration_range ( 10u )

// 75% zmiany w stronę osiągnięcia średniej sąsiadów //
#define temp_adaptation_to_neighbors_factor ( 0.75 )
#define soft_max_param ( 0.1 )

#define constant_cooling ( 0.1 )
#define sides_heating ( 20 )

// temperatura przy której prawdopodobieństwo zmiany ID = 50% //
#define temp_threshold_for_id_change ( ZERO_CELC_IN_KELV )

// / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / //

#define TEMP_CHANGES

// #define ID_CHANGES_MAX_TEMP
#define ID_CHANGES_SOFTMAXING_SUMMED_TEMP
