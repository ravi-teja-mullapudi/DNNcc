#include "NDArray.h"
#include "Op.h"

void conv2d_forward_ref(int batch_size,
                        int input_channels,
                        int input_height,
                        int input_width,
                        int output_channels,
                        int filter_height,
                        int filter_width,
                        int stride_h,
                        int stride_w,
                        NDArray<float>& input,
                        NDArray<float>& weights,
                        NDArray<float>& output);

void pool2d_forward_ref(int batch_size,
                        int input_channels,
                        int input_height,
                        int input_width,
                        int pool_width,
                        int pool_height,
                        int stride_h,
                        int stride_w,
                        PoolType pool_type,
                        NDArray<float>& input,
                        NDArray<float>& output);
