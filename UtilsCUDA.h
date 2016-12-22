#pragma once

#include <cstdio>
#include <vector>
#include <chrono>
#include <assert.h>

#define gpu_err_chk(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

template <typename F>
double benchmark(int samples, int iterations, F op) {
    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < samples; i++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < iterations; j++) {
            op();
        }
        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1e6;
        if (dt < best) best = dt;
    }
    return best / iterations;
}

struct LaunchConfig {
    dim3 block_config;
    dim3 thread_config;
};

LaunchConfig get_tile_launch_config(std::vector<int> tile_sizes,
                                    std::vector<int> dim_sizes) {
    assert(tile_sizes.size() == dim_sizes.size());
    assert(tile_sizes.size() <= 3);

    LaunchConfig l;
    switch(tile_sizes.size()) {
        case 1:
            l.thread_config.x = tile_sizes[0];
            l.block_config.x = std::ceil((float)dim_sizes[0]/tile_sizes[0]);
            break;
        case 2:
            l.thread_config.x = tile_sizes[0];
            l.thread_config.y = tile_sizes[1];
            l.block_config.x = std::ceil((float)dim_sizes[0]/tile_sizes[0]);
            l.block_config.y = std::ceil((float)dim_sizes[1]/tile_sizes[1]);
            break;
        case 3:
            l.thread_config.x = tile_sizes[0];
            l.thread_config.y = tile_sizes[1];
            l.thread_config.z = tile_sizes[2];
            l.block_config.x = std::ceil((float)dim_sizes[0]/tile_sizes[0]);
            l.block_config.y = std::ceil((float)dim_sizes[1]/tile_sizes[1]);
            l.block_config.z = std::ceil((float)dim_sizes[2]/tile_sizes[2]);
            break;
        default:
            fprintf(stderr, "Incorrect parameters for launch config\n");
    }

    return l;
}
