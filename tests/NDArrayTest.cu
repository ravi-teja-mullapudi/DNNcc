#include "NDArrayCUDA.h"
#include <iostream>

__global__ void copy_test(int* input, int xsize, int ysize, int* output) {
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;

    int x = tile_x * blockDim.x + threadIdx.x;
    int y = tile_y * blockDim.y + threadIdx.y;

    int offset = x * ysize + y;
    if (x < xsize && y < ysize) {
        output[offset] = input[offset];
    }
}

int main() {

    NDArrayCUDA<int> input({76, 899});
    NDArrayCUDA<int> output({76, 899});

    input.device_allocate();
    output.device_allocate();

    input.initialize(5);
    input(0, 1) = 10;
    input(1, 0) = 1000;
    input(75, 888) = 7;
    input(0, 0) = 9;

    input.copy_to_device();

    LaunchConfig l = get_tile_launch_config({8, 8},
                                {input.dim_sizes[0], input.dim_sizes[1]});

    double time = benchmark(1, 1, [&]() {
    copy_test<<<l.block_config, l.thread_config>>>(input.dev_alloc.get(),
                                                   input.dim_sizes[0],
                                                   input.dim_sizes[1],
                                                   output.dev_alloc.get());
    });

    std::cout << "Time: " << time << "ms" << std::endl;
    output.copy_from_device();

    for (int x = 0; x < input.dim_sizes[0];  x++) {
        for (int y = 0; y < input.dim_sizes[1]; y++) {
            if (output(x, y) != input(x, y)) {
                std::cerr << "Test failed" << std::endl;
                exit(-1);
            }
        }
    }
    std::cout << "Test passed" << std::endl;
}
