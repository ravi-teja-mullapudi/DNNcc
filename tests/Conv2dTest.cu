#include <iostream>
#include <assert.h>
#include "NDArray.h"
#include "Utils.h"
#include "UtilsCUDNN.h"
#include "OpShapes.h"

float benchmark_conv2d_cudnn(ConvLayerShape& shape) {
    int batch_size = shape.batch_size;
    int input_channels = shape.input_channels;
    int input_height = shape.input_height;
    int input_width = shape.input_width;
    int output_channels = shape.output_channels;

    int filter_width = shape.filter_width;
    int filter_height = shape.filter_height;
    int stride = shape.stride;

    // Host buffers for holding inputs and outputs to the
    // conv layer
    NDArray<float> input({input_channels, input_height,
                          input_width, batch_size});
    NDArray<float> weights({output_channels, input_channels,
                            filter_height, filter_width});

    int pad_w = (filter_width - 1)/2;
    int pad_h = (filter_height - 1)/2;

    int output_width = (1 + (input_width + 2 * pad_w - filter_width)/stride);
    int output_height = (1 + (input_height + 2 * pad_h - filter_height)/stride);

    NDArray<float> output({output_channels, output_height,
                           output_width, batch_size});

    // Create corresponding int arrays on the GPU.
    input.device_allocate();
    weights.device_allocate();
    output.device_allocate();

    // Copy input data to array on GPU.
    input.copy_to_device();
    weights.copy_to_device();

    CudnnConv2d cnn(input_width, input_height, input_channels, batch_size,
                    output_channels, filter_width, filter_height, pad_w, pad_h,
                    stride, stride);

    //std::string fwd_algo_s = cnn.get_fwd_algo_string();

    float time = benchmark(5, 1, [&]() {
        cnn.forward(input, weights, output);
    });

    output.copy_from_device();

    return time * 1000;
}

int main() {

    std::vector<ConvLayerShape> conv_shapes;
    conv_shapes.push_back(ConvLayerShape(64, 56, 56, 64, 64, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 28, 28, 192, 64, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 28, 28, 192, 96, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 224, 224, 3, 64, 7, 7, 2));
    conv_shapes.push_back(ConvLayerShape(64, 56, 56, 64, 192, 3, 3, 1));
    conv_shapes.push_back(ConvLayerShape(16, 112, 112, 64, 64, 3, 3, 1));
    conv_shapes.push_back(ConvLayerShape(64, 28, 28, 96, 128, 3, 3, 1));
    conv_shapes.push_back(ConvLayerShape(64, 28, 28, 128, 192, 3, 3, 1));
    conv_shapes.push_back(ConvLayerShape(64, 28, 28, 32, 96, 5, 5, 1));
    conv_shapes.push_back(ConvLayerShape(64, 14, 14, 96, 208, 3, 3, 1));
    conv_shapes.push_back(ConvLayerShape(64, 14, 14, 112, 224, 3, 3, 1));
    conv_shapes.push_back(ConvLayerShape(64, 28, 28, 192, 16, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 28, 28, 192, 32, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 28, 28, 256, 128, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 28, 28, 256, 32, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 28, 28, 256, 64, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 14, 14, 480, 192, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 14, 14, 480, 96, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 14, 14, 480, 16, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 14, 14, 480, 64, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 14, 14, 512, 160, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 14, 14, 512, 112, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 14, 14, 512, 24, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 14, 14, 24, 64, 5, 5, 1));
    conv_shapes.push_back(ConvLayerShape(64, 14, 14, 512, 64, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 14, 14, 512, 128, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 14, 14, 128, 256, 3, 3, 1));
    conv_shapes.push_back(ConvLayerShape(64, 14, 14, 512, 24, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 14, 14, 24, 64, 5, 5, 1));
    conv_shapes.push_back(ConvLayerShape(64, 14, 14, 512, 64, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 14, 14, 512, 112, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 14, 14, 512, 144, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 14, 14, 114, 228, 3, 3, 1));
    conv_shapes.push_back(ConvLayerShape(64, 14, 14, 512, 32, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 14, 14, 32, 64, 5, 5, 1));
    conv_shapes.push_back(ConvLayerShape(64, 14, 14, 528, 256, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 14, 14, 528, 160, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 14, 14, 160, 320, 3, 3, 1));
    conv_shapes.push_back(ConvLayerShape(64, 14, 14, 528, 32, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 14, 14, 32, 128, 5, 5, 1));
    conv_shapes.push_back(ConvLayerShape(64, 14, 14, 512, 128, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 7, 7, 832, 256, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 7, 7, 832, 160, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 7, 7, 160, 320, 3, 3, 1));
    conv_shapes.push_back(ConvLayerShape(64, 7, 7, 832, 32, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 7, 7, 32, 128, 5, 5, 1));
    conv_shapes.push_back(ConvLayerShape(64, 7, 7, 832, 128, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 7, 7, 832, 384, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 7, 7, 832, 192, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 7, 7, 192, 384, 3, 3, 1));
    conv_shapes.push_back(ConvLayerShape(64, 7, 7, 832, 48, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(64, 7, 7, 48, 128, 5, 5, 1));
    conv_shapes.push_back(ConvLayerShape(64, 7, 7, 832, 128, 1, 1, 1));

    for (auto &s: conv_shapes) {
        std::cout << benchmark_conv2d_cudnn(s) << std::endl;
    }

    return 0;
}
