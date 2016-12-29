#include "OpRef.h"
void affine_forward_ref(std::shared_ptr<AffineOp> op,
                        NDArray<float>& input,
                        NDArray<float>& output) {

}

void conv2d_forward_ref(std::shared_ptr<Conv2dOp> op,
                        NDArray<float>& input,
                        NDArray<float>& output) {

    int batch_size = op->batch_size;
    int input_channels = op->input_channels;
    int input_height = op->input_height;
    int input_width = op->input_width;
    int output_channels = op->output_channels;
    int filter_width = op->filter_width;
    int filter_height = op->filter_height;
    int stride_h = op->stride_h;
    int stride_w = op->stride_w;
    int pad_w = op->pad_w;
    int pad_h = op->pad_h;
    int output_height = op->output_height;
    int output_width = op->output_width;

    NDArray<float>& weights = get_ndarray<float>(op->params[0]);
    NDArray<float>& bias = get_ndarray<float>(op->params[1]);

    for (int b = 0; b < batch_size; b++) {
        for(int out_c = 0; out_c < output_channels; out_c++) {
            for(int out_h = 0; out_h < output_height; out_h++) {
                for(int out_w = 0; out_w < output_width; out_w++) {
                    float val = 0.0f;
                    for (int in_c = 0; in_c < input_channels; in_c++) {
                        for (int f_h = 0; f_h < filter_height; f_h++) {
                            for (int f_w = 0; f_w < filter_width; f_w++) {
                                int in_h = out_h * stride_h + f_h - pad_h;
                                int in_w = out_w * stride_w + f_w - pad_w;
                                // Constant boundary condition 0.0f
                                float in_val = (in_w >= 0 && in_w < input_width) &&
                                               (in_h >= 0 && in_h < input_height) ?
                                               input(b, in_c, in_h, in_w) : 0.0f;
                                val += weights(out_c, in_c, f_h, f_w) * in_val;
                            }
                        }
                    }
                    output(b, out_c, out_h, out_w) = val + bias(out_c);
                }
            }
        }
    }
}

void pool2d_forward_ref(std::shared_ptr<Pool2dOp> op,
                        NDArray<float>& input,
                        NDArray<float>& output) {

    PoolType pool_type = op->pool_type;
    int batch_size = op->batch_size;
    int input_channels = op->input_channels;
    int input_height = op->input_height;
    int input_width = op->input_width;
    int pool_width = op->pool_width;
    int pool_height = op->pool_height;
    int stride_h = op->stride_h;
    int stride_w = op->stride_w;
    int pad_w = op->pad_w;
    int pad_h = op->pad_h;
    int output_width = op->output_width;
    int output_height = op->output_height;

    for(int b = 0; b < batch_size; b++) {
        for(int ch = 0; ch < input_channels; ch++) {
            for(int h = 0; h < output_height; h++) {
                for(int w = 0; w < output_width; w++) {
                    if (pool_type == PoolType::AVG) {
                        // TODO: CUDNN has other modes where the boundary values
                        // are not taken into account when doing the average.
                        float sum = 0;
                        int num = 0;
                        for(int k_h = 0; k_h < pool_height; k_h++) {
                            for(int k_w = 0; k_w < pool_width; k_w++) {
                                int in_w = w * stride_w + k_w - pad_w;
                                int in_h = h * stride_h + k_h - pad_h;
                                float in_val = (in_w >= 0 && in_w < input_width) &&
                                               (in_h >= 0 && in_h < input_height) ?
                                               input(b, ch, in_h, in_w) : 0.0f;
                                sum += in_val;
                                num++;
                            }
                        }
                        output(b, ch, h, w) = sum/num;
                    } else if (pool_type == PoolType::MAX) {
                        // TODO: CUDNN has other modes where the boundary values
                        // are not taken into account when doing the max.
                        float max = input(b, ch, h * stride_h, w * stride_w);
                        for(int k_h = 0; k_h < pool_height; k_h++) {
                            for(int k_w = 0; k_w < pool_width; k_w++) {
                                int in_w = w * stride_w + k_w - pad_w;
                                int in_h = h * stride_h + k_h - pad_h;
                                float in_val = (in_w >= 0 && in_w < input_width) &&
                                               (in_h >= 0 && in_h < input_height) ?
                                               input(b, ch, in_h, in_w) : 0.0f;
                                if (in_val > max) {
                                    max = in_val;
                                }
                            }
                        }
                        output(b, ch, h, w) = max;
                    }
                }
            }
        }
    }
}

void relu_forward_ref(std::shared_ptr<ReLUOp> op,
                      NDArray<float>& input,
                      NDArray<float>& output) {
}

void softmax_forward_ref(std::shared_ptr<SoftMaxOp> op,
                         NDArray<float>& input,
                         NDArray<float>& output) {
}

void lrn_forward_ref(std::shared_ptr<LRNOp> op,
                     NDArray<float>& input,
                     NDArray<float>& output) {
}

void concat_forward_ref(std::shared_ptr<ConcatOp> op,
                        std::vector<NDArray<float>>& inputs,
                        NDArray<float>& output) {
}

void flatten_forward_ref(std::shared_ptr<FlattenOp> op,
                         NDArray<float>& input,
                         NDArray<float>& output) {
}

void data_forward_ref(std::shared_ptr<DataOp> op,
                      NDArray<float>& input,
                      NDArray<float>& output) {
}
