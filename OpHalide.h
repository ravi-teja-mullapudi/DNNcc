#pragma once

#include <string>
#include <vector>
#include "Halide.h"
#include "OpImpl.h"
#include "Op.h"

using namespace Halide;

// Sanity check to make sure the halide function is defined.
void check_defined(Func f);

class OpHalideImpl {
    public:
    Func output;
    // Ordered list of learnable parameters of the op.
    std::vector<ImageParam> params;
    // Ordered list of learnable parameters of the op.
    std::vector<Func> param_grads;
    // Ordered list of gradient inputs to the op.
    std::vector<Func> input_grads;
};

void sum_forward_halide(std::string name,
                        std::shared_ptr<SumOp> op,
                        std::vector<Func> inputs,
                        std::shared_ptr<OpHalideImpl> op_impl,
                        TargetArch arch);

void affine_forward_halide(std::string name,
                           std::shared_ptr<AffineOp> op,
                           Func input,
                           std::shared_ptr<OpHalideImpl> op_impl,
                           TargetArch arch);

void conv2d_forward_halide(std::string name,
                           std::shared_ptr<Conv2dOp> op,
                           Func input,
                           std::shared_ptr<OpHalideImpl> op_impl,
                           TargetArch arch);

void pool2d_forward_halide(std::string name,
                           std::shared_ptr<Pool2dOp> op,
                           Func input,
                           std::shared_ptr<OpHalideImpl> op_impl,
                           TargetArch arch);

void relu_forward_halide(std::string name,
                         std::shared_ptr<ReLUOp> op,
                         Func input,
                         std::shared_ptr<OpHalideImpl> op_impl,
                         TargetArch arch);

void softmax_forward_halide(std::string name,
                            std::shared_ptr<SoftMaxOp> op,
                            Func input,
                            std::shared_ptr<OpHalideImpl> op_impl,
                            TargetArch arch);

void lrn_forward_halide(std::string name,
                        std::shared_ptr<LRNOp> op,
                        Func input,
                        std::shared_ptr<OpHalideImpl> op_impl,
                        TargetArch arch);

void concat_forward_halide(std::string name,
                           std::shared_ptr<ConcatOp> op,
                           std::vector<Func> inputs,
                           std::shared_ptr<OpHalideImpl> op_impl,
                           TargetArch arch);

void flatten_forward_halide(std::string name,
                           std::shared_ptr<FlattenOp> op,
                           Func input,
                           std::shared_ptr<OpHalideImpl> op_impl,
                           TargetArch arch);

void data_forward_halide(std::string name,
                         std::shared_ptr<DataOp> op,
                         Func input,
                         std::shared_ptr<OpHalideImpl> op_impl,
                         TargetArch arch);

Buffer<> get_halide_buffer(NDArray_t& arr, DataType type);
