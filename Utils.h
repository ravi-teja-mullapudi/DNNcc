#include <cmath>
#include <random>

template<typename T>
inline bool is_nearly_equal(T x, T y)
{
    const T epsilon = 1e-5;
    return std::abs(x - y) <= epsilon * std::abs(x);
}

template<typename T>
struct GaussianGenerator {
    const T mean;
    const T std_dev;

    std::random_device rd;
    std::mt19937 e;
    std::normal_distribution<T> d;

    GaussianGenerator(T _mean, T _std_dev) :
                      mean(_mean), std_dev(_std_dev) {
        e = std::mt19937(rd());
        d = std::normal_distribution<T>(mean, std_dev);
    }

    T operator()() {
        return d(e);
    }
};
