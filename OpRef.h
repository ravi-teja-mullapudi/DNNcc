#include "NDArray.h"
#include "Op.h"

void affine_forward_ref(std::shared_ptr<AffineOp> op,
                        NDArray<float>& input,
                        NDArray<float>& output);

void conv2d_forward_ref(std::shared_ptr<Conv2dOp> op,
                        NDArray<float>& input,
                        NDArray<float>& output);

void pool2d_forward_ref(std::shared_ptr<Pool2dOp> op,
                        NDArray<float>& input,
                        NDArray<float>& output);

void relu_forward_ref(std::shared_ptr<ReLUOp> op,
                      NDArray<float>& input,
                      NDArray<float>& output);

void softmax_forward_ref(std::shared_ptr<SoftMaxOp> op,
                         NDArray<float>& input,
                         NDArray<float>& output);

void lrn_forward_ref(std::shared_ptr<LRNOp> op,
                     NDArray<float>& input,
                     NDArray<float>& output);

void concat_forward_ref(std::shared_ptr<ConcatOp> op,
                        std::vector<NDArray<float>>& inputs,
                        NDArray<float>& output);

void flatten_forward_ref(std::shared_ptr<FlattenOp> op,
                         NDArray<float>& input,
                         NDArray<float>& output);

void data_forward_ref(std::shared_ptr<DataOp> op,
                      NDArray<float>& input,
                      NDArray<float>& output);
