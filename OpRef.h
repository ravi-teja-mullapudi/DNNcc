#include "NDArray.h"
#include "Op.h"

template <typename T>
void sum_forward_ref(std::shared_ptr<AffineOp> op,
                     std::vector<NDArray<T>>& inputs,
                     NDArray<T>& output);

template <typename T>
void affine_forward_ref(std::shared_ptr<AffineOp> op,
                        NDArray<T>& input,
                        NDArray<T>& output);

template <typename T>
void conv2d_forward_ref(std::shared_ptr<Conv2dOp> op,
                        NDArray<T>& input,
                        NDArray<T>& output);

template <typename T>
void pool2d_forward_ref(std::shared_ptr<Pool2dOp> op,
                        NDArray<T>& input,
                        NDArray<T>& output);

template <typename T>
void relu_forward_ref(std::shared_ptr<ReLUOp> op,
                      NDArray<T>& input,
                      NDArray<T>& output);

template <typename T>
void softmax_forward_ref(std::shared_ptr<SoftMaxOp> op,
                         NDArray<T>& input,
                         NDArray<T>& output);

template <typename T>
void lrn_forward_ref(std::shared_ptr<LRNOp> op,
                     NDArray<T>& input,
                     NDArray<T>& output);

template <typename T>
void concat_forward_ref(std::shared_ptr<ConcatOp> op,
                        std::vector<NDArray<T>>& inputs,
                        NDArray<T>& output);
template <typename T>
void flatten_forward_ref(std::shared_ptr<FlattenOp> op,
                         NDArray<T>& input,
                         NDArray<T>& output);
template <typename T>
void data_forward_ref(std::shared_ptr<DataOp> op,
                      NDArray<T>& input,
                      NDArray<T>& output);
