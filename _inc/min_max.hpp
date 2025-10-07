#pragma once
#include <limits>

template<typename T>
class MinMax
{
    T min;
    T max;

public:
    MinMax(T _min = std::numeric_limits<T>::max(), T _max = std::numeric_limits<T>::lowest()) : min(_min), max(_max) {}

    void update(const T& value)
    {
        min = std::min(min, value);
        max = std::max(max, value);
    }

    T get_min() const { return min; }
    T get_max() const { return max; }
};

template<typename T>
class Average
{
    T sum;
    size_t count;

public:
    Average() { init(); }

    void init()
    {
        sum = {};
        count = 0;
    }

    void add(const T& value)
    {
        sum += value;
        count++;
    }

    T get() const { return count > 0 ? (sum / count) : T{}; }
};