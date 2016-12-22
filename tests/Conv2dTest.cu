#include <iostream>
#include <assert.h>
#include "NDArrayCUDA.h"
#include "Utils.h"
#include "UtilsCUDNN.h"
#include "OpShapes.h"
#include "OpRef.h"

void test_conv2d_cudnn(ConvLayerShape& shape) {
    int batch_size = shape.batch_size;
    int input_channels = shape.input_channels;
    int input_height = shape.input_height;
    int input_width = shape.input_width;
    int output_channels = shape.output_channels;

    int filter_width = shape.filter_width;
    int filter_height = shape.filter_height;
    int stride = shape.stride;

    NDArrayCUDA<float> input({batch_size, input_channels,
                              input_height, input_width});
    NDArrayCUDA<float> weights({output_channels, input_channels,
                                filter_height, filter_width});

    int pad_w = (filter_width - 1)/2;
    int pad_h = (filter_height - 1)/2;

    int output_width = (1 + (input_width + 2 * pad_w - filter_width)/stride);
    int output_height = (1 + (input_height + 2 * pad_h - filter_height)/stride);

    NDArrayCUDA<float> output_ref({batch_size, output_channels,
                                   output_height, output_width});
    NDArrayCUDA<float> output_cudnn({batch_size, output_channels,
                                     output_height, output_width});

    // Create corresponding int arrays on the GPU.
    input.device_allocate();
    weights.device_allocate();
    output_ref.device_allocate();
    output_cudnn.device_allocate();

    // Initialize the input data on the CPU.
    GaussianGenerator<float> rgen(1.0f, 0.1f);
    input.initialize(rgen);
    weights.initialize(rgen);

    // Copy input data to array on GPU.
    input.copy_to_device();
    weights.copy_to_device();

    CudnnConv2d cnn(input_width, input_height, input_channels, batch_size,
                    output_channels, filter_width, filter_height, pad_w, pad_h,
                    stride, stride);

    cnn.forward(input, weights, output_cudnn);

    output_cudnn.copy_from_device();

    conv2d_forward_ref(batch_size, input_channels, input_height, input_width,
                       output_channels, filter_height, filter_width, stride, stride,
                       input, weights, output_ref);

    for (int b = 0; b < batch_size; b++) {
        for (int out_c = 0; out_c < output_channels; out_c++) {
            for (int out_h = 0; out_h < output_height; out_h++) {
                for (int out_w = 0; out_w < output_width; out_w++) {
                    if (!is_nearly_equal(output_cudnn(b, out_c, out_h, out_w),
                                         output_ref(b, out_c, out_h, out_w))) {
                        std::cerr <<
                            "CUDNN gpu results do not match with reference" << std::endl;
                        exit(-1);
                    }
                }
            }
        }
    }

    std::cout << "Passed:" << shape << std::endl;
}

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
    NDArrayCUDA<float> input({input_channels, input_height,
                              input_width, batch_size});
    NDArrayCUDA<float> weights({output_channels, input_channels,
                                filter_height, filter_width});

    int pad_w = (filter_width - 1)/2;
    int pad_h = (filter_height - 1)/2;

    int output_width = (1 + (input_width + 2 * pad_w - filter_width)/stride);
    int output_height = (1 + (input_height + 2 * pad_h - filter_height)/stride);

    NDArrayCUDA<float> output({output_channels, output_height,
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
    // TODO: Make this an option
    int batch_size = 64;

    conv_shapes.push_back(ConvLayerShape(batch_size, 56, 56, 64, 64, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 28, 28, 192, 64, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 28, 28, 192, 96, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 224, 224, 3, 64, 7, 7, 2));
    conv_shapes.push_back(ConvLayerShape(batch_size, 56, 56, 64, 192, 3, 3, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 112, 112, 64, 64, 3, 3, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 28, 28, 96, 128, 3, 3, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 28, 28, 128, 192, 3, 3, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 28, 28, 32, 96, 5, 5, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 14, 14, 96, 208, 3, 3, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 14, 14, 112, 224, 3, 3, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 28, 28, 192, 16, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 28, 28, 192, 32, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 28, 28, 256, 128, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 28, 28, 256, 32, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 28, 28, 256, 64, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 14, 14, 480, 192, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 14, 14, 480, 96, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 14, 14, 480, 16, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 14, 14, 480, 64, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 14, 14, 512, 160, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 14, 14, 512, 112, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 14, 14, 512, 24, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 14, 14, 24, 64, 5, 5, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 14, 14, 512, 64, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 14, 14, 512, 128, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 14, 14, 128, 256, 3, 3, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 14, 14, 512, 24, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 14, 14, 24, 64, 5, 5, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 14, 14, 512, 64, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 14, 14, 512, 112, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 14, 14, 512, 144, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 14, 14, 114, 228, 3, 3, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 14, 14, 512, 32, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 14, 14, 32, 64, 5, 5, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 14, 14, 528, 256, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 14, 14, 528, 160, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 14, 14, 160, 320, 3, 3, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 14, 14, 528, 32, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 14, 14, 32, 128, 5, 5, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 14, 14, 512, 128, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 7, 7, 832, 256, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 7, 7, 832, 160, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 7, 7, 160, 320, 3, 3, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 7, 7, 832, 32, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 7, 7, 32, 128, 5, 5, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 7, 7, 832, 128, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 7, 7, 832, 384, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 7, 7, 832, 192, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 7, 7, 192, 384, 3, 3, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 7, 7, 832, 48, 1, 1, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 7, 7, 48, 128, 5, 5, 1));
    conv_shapes.push_back(ConvLayerShape(batch_size, 7, 7, 832, 128, 1, 1, 1));

    /*
    // TODO: Make this an option
    for (auto &s: conv_shapes) {
        test_conv2d_cudnn(s);
    }
    */

    for (auto &s: conv_shapes) {
        std::cout << benchmark_conv2d_cudnn(s) << std::endl;
    }

    return 0;
}
